import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import yaml
from typing import Dict, Tuple, List
import time

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int, layers: int = 4, 
                 dropout: float = 0.0, activation: str = 'gelu'):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.layers = layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.gelu
        
        # Enhanced input projection with normalization
        self.fc0 = nn.Sequential(
            nn.Linear(3, self.width),
            nn.LayerNorm(self.width)  # Add normalization
        )
        
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2) 
            for _ in range(self.layers)
        ])
        
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) 
            for _ in range(self.layers)
        ])
        
        # Add layer normalization for better training stability
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.width) for _ in range(self.layers)
        ])
        
        # Enhanced output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, x_dim, y_dim, 3) where 3 = (x_coord, y_coord, rhs)
        x = self.fc0(x)  # (batch, x_dim, y_dim, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, x_dim, y_dim)
        
        for i in range(self.layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            
            # Apply normalization and activation
            x = x.permute(0, 2, 3, 1)  # (batch, x_dim, y_dim, width)
            x = self.norm_layers[i](x)
            if i < self.layers - 1:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            x = x.permute(0, 3, 1, 2)  # (batch, width, x_dim, y_dim)
        
        x = x.permute(0, 2, 3, 1)  # (batch, x_dim, y_dim, width)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)  # (batch, x_dim, y_dim, 1)
        
        return x.squeeze(-1)  # (batch, x_dim, y_dim)

def rhs_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side function for the Poisson equation
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    f_temp = -2.0 * (omega_x**2) * (np.sin(omega_x * x) * np.sin(omega_y * y))
    return f_temp

def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Exact solution for the Poisson equation
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    return np.sin(omega_x * x) * np.sin(omega_y * y)

def generate_data(n_samples: int, nx: int, ny: int, 
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for the 2D Poisson equation
    """
    # Create coordinate grids
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Generate input features and solutions
    inputs = []
    solutions = []
    
    for _ in range(n_samples):
        # For this example, we use the same RHS function
        # In practice, you might vary parameters or use different RHS functions
        rhs_vals = rhs_function(X, Y)
        exact_sol = exact_solution(X, Y)
        
        # Create input tensor: (x_coord, y_coord, rhs_value)
        input_tensor = np.stack([X, Y, rhs_vals], axis=-1)
        
        inputs.append(input_tensor)
        solutions.append(exact_sol)
    
    inputs = np.array(inputs)
    solutions = np.array(solutions)
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(solutions, dtype=torch.float32)

def relative_l2_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute relative L2 error
    """
    return torch.norm(pred - true) / torch.norm(true)

class PoissonFNOTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FNO2d(
            modes1=config.get('modes1', 12),
            modes2=config.get('modes2', 12),
            width=config.get('width', 32),
            layers=config.get('layers', 4)
        ).to(self.device)
        
        # Enhanced optimizer with weight decay
        self.optimizer = torch.optim.AdamW(  # AdamW often works better
            self.model.parameters(),
            lr=config['learning_rate']['initial_learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        if config['learning_rate']['use_lr_scheduler']:
            # Using CosineAnnealingLR for smoother decay
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = None
            
        # Enhanced loss function with relative error component
        self.loss_fn = nn.MSELoss()
        self.gradient_clipping = config.get('gradient_clipping', None)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Enhanced training
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 200
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_rel_error = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Combined loss: MSE + relative error
                mse_loss = self.loss_fn(outputs, targets)
                rel_loss = torch.mean(torch.norm(outputs - targets, dim=(1,2)) / 
                                    torch.norm(targets, dim=(1,2)))
                loss = mse_loss + 0.1 * rel_loss  # Weight relative error
                
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.gradient_clipping)
                
                self.optimizer.step()
                
                train_loss += mse_loss.item()
                train_rel_error += rel_loss.item()
            
            train_loss /= len(train_loader)
            train_rel_error /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss = float('inf')
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_rel_error = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        val_loss += self.loss_fn(outputs, targets).item()
                        val_rel_error += torch.mean(torch.norm(outputs - targets, dim=(1,2)) / 
                                                  torch.norm(targets, dim=(1,2))).item()
                
                val_loss /= len(val_loader)
                val_rel_error /= len(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_fno_model.pth')
                else:
                    patience_counter += 1
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print progress with more details
            if (epoch + 1) % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if val_loader is not None:
                    print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                          f'Train Loss: {train_loss:.8f}, Train Rel: {train_rel_error:.6f}, '
                          f'Val Loss: {val_loss:.8f}, Val Rel: {val_rel_error:.6f}, '
                          f'LR: {current_lr:.2e}')
                else:
                    print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                          f'Train Loss: {train_loss:.8f}, Train Rel: {train_rel_error:.6f}, '
                          f'LR: {current_lr:.2e}')
            
            # Early stopping check
            if patience_counter >= patience and epoch > 500:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_fno_model.pth'))
            print("Loaded best model from validation")
        
        return train_losses, val_losses
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu()

def plot_results(x_coords: np.ndarray, y_coords: np.ndarray, 
                true_solution: np.ndarray, predicted_solution: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # True solution
    im1 = axes[0].contourf(x_coords, y_coords, true_solution, levels=50, cmap='jet')
    axes[0].set_title('True Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Predicted solution
    im2 = axes[1].contourf(x_coords, y_coords, predicted_solution, levels=50, cmap='jet')
    axes[1].set_title('FNO Prediction')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(true_solution - predicted_solution)
    im3 = axes[2].contourf(x_coords, y_coords, error, levels=50, cmap='jet')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    plt.savefig('fno_results.png')
    plt.close(fig)

def main():
    """
    Main function to run FNO training for 2D Poisson equation
    """
    # Enhanced Configuration for better accuracy
    config = {
        # Increase Fourier modes for better frequency resolution
        'modes1': 20,           # Was 12, now 20 (capture more frequencies)
        'modes2': 20,           # Was 12, now 20
        
        # Increase network capacity
        'width': 64,            # Was 32, now 64 (more channels)
        'layers': 6,            # Was 4, now 6 (deeper network)
        
        # More training
        'epochs': 2000,         # Was 1000, now 2000
        'batch_size': 8,        # Smaller batch for more stable gradients
        
        # Better learning rate schedule
        'learning_rate': {
            'initial_learning_rate': 0.002,  # Slightly higher initial LR
            'use_lr_scheduler': True,
            'decay_steps': 300,              # Decay less frequently
            'decay_rate': 0.98               # Gentler decay
        },
        
        # Higher resolution for more data
        'geometry': {
            'x_min': 0, 'x_max': 1,
            'y_min': 0, 'y_max': 1,
            'nx': 128,   # Was 64, now 128 (higher resolution)
            'ny': 128    # Was 64, now 128
        },
        
        # Additional tuning parameters
        'weight_decay': 1e-5,   # Regularization
        'gradient_clipping': 1.0,  # Prevent exploding gradients
    }
    
    # Generate more training data for better generalization
    print("Generating training data...")
    train_inputs, train_targets = generate_data(
        n_samples=200,  # Was 100, now 200
        nx=config['geometry']['nx'],
        ny=config['geometry']['ny']
    )
    
    val_inputs, val_targets = generate_data(
        n_samples=40,   # Was 20, now 40
        nx=config['geometry']['nx'], 
        ny=config['geometry']['ny']
    )
    
    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize trainer and train model
    trainer = PoissonFNOTrainer(config)
    
    print("Starting training...")
    start_time = time.time()
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test the model
    test_input, test_target = generate_data(
        n_samples=1,
        nx=config['geometry']['nx'],
        ny=config['geometry']['ny']
    )
    
    prediction = trainer.predict(test_input)
    
    # Compute error
    rel_error = relative_l2_error(prediction[0], test_target[0])
    print(f"Relative L2 error: {rel_error:.6f}")
    
    # Create coordinate grids for plotting
    x = np.linspace(0, 1, config['geometry']['nx'])
    y = np.linspace(0, 1, config['geometry']['ny'])
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Plot results
    plot_results(X, Y, test_target[0].numpy(), prediction[0].numpy())
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-100:], label='Train Loss (last 100)')
    plt.plot(val_losses[-100:], label='Val Loss (last 100)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History (Final Epochs)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('training_history.png')

if __name__ == "__main__":
    main()