import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

class PoissonDataset(Dataset):
    """Dataset for 2D Poisson equation problems"""
    def __init__(self, n_samples=1000, grid_size=64):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.data = []
        self.solutions = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{n_samples} samples")
            
            # Generate random source term f(x,y)
            source = self.generate_random_source()
            
            # Solve using FEM
            solution = self.solve_fem(source)
            
            self.data.append(torch.FloatTensor(source))
            self.solutions.append(torch.FloatTensor(solution))
    
    def generate_random_source(self):
        """Generate random source term with smooth spatial variation"""
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Random combination of Gaussian sources
        n_sources = np.random.randint(1, 4)
        source = np.zeros((self.grid_size, self.grid_size))
        
        for _ in range(n_sources):
            # Random center and width
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            sigma = np.random.uniform(0.05, 0.2)
            amplitude = np.random.uniform(-2, 2)
            
            # Add Gaussian source
            source += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        return source
    
    def solve_fem(self, source):
        """Solve 2D Poisson equation using Finite Element Method"""
        n = self.grid_size
        h = 1.0 / (n - 1)
        
        # Create 2D finite difference matrix (approximates FEM for regular grid)
        # -∇²u = f with Dirichlet boundary conditions u = 0
        
        # Interior points
        N = (n-2) * (n-2)  # Number of interior points
        
        # Build the system matrix A
        main_diag = 4 * np.ones(N)
        off_diag = -1 * np.ones(N-1)
        far_diag = -1 * np.ones(N-(n-2))
        
        # Handle boundary effects in off-diagonal
        for i in range(1, N):
            if i % (n-2) == 0:  # Right boundary of interior domain
                off_diag[i-1] = 0
        
        A = diags([far_diag, off_diag, main_diag, off_diag, far_diag], 
                  [-(n-2), -1, 0, 1, n-2], format='csr')
        A = A / (h**2)
        
        # Right hand side (source term at interior points)
        b = source[1:-1, 1:-1].flatten()
        
        # Solve the linear system
        u_interior = spsolve(A, b)
        
        # Reconstruct full solution with boundary conditions
        u_full = np.zeros((n, n))
        u_full[1:-1, 1:-1] = u_interior.reshape(n-2, n-2)
        
        return u_full
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.solutions[idx]

class SpectralConv2d(nn.Module):
    """Spectral convolution layer for FNO"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D problems"""
    def __init__(self, modes1=16, modes2=16, width=64, n_layers=4):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        self.fc0 = nn.Linear(1, self.width)  # Input: source term
        
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.activation = F.gelu
    
    def forward(self, x):
        # x shape: (batch, height, width)
        x = x.unsqueeze(-1)  # Add channel dimension
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = self.activation(x)
        
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.squeeze(-1)  # Remove channel dimension
        
        return x

def train_fno(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (source, target) in enumerate(train_loader):
            source, target = source.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for source, target in val_loader:
                source, target = source.to(device), target.to(device)
                output = model(source)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_methods(test_dataset, fno_model, grid_size=64):
    """Compare FEM and FNO performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fno_model = fno_model.to(device)
    fno_model.eval()
    
    n_test = len(test_dataset)
    fem_times = []
    fno_times = []
    fem_errors = []
    fno_errors = []
    
    print(f"Evaluating on {n_test} test samples...")
    
    for i in range(min(n_test, 100)):  # Test on first 100 samples
        source, fem_solution = test_dataset[i]
        
        # Time FEM solution
        start_time = time.time()
        fem_pred = test_dataset.solve_fem(source.numpy())
        fem_time = time.time() - start_time
        fem_times.append(fem_time)
        
        # Time FNO solution
        start_time = time.time()
        with torch.no_grad():
            source_tensor = source.unsqueeze(0).to(device)
            fno_pred = fno_model(source_tensor).cpu().numpy()[0]
        fno_time = time.time() - start_time
        fno_times.append(fno_time)
        
        # Compute errors (using FEM as ground truth)
        fem_error = np.mean((fem_pred - fem_solution.numpy())**2)
        fno_error = np.mean((fno_pred - fem_solution.numpy())**2)
        
        fem_errors.append(fem_error)
        fno_errors.append(fno_error)
        
        if i % 20 == 0:
            print(f"Evaluated {i+1}/100 samples")
    
    return {
        'fem_times': fem_times,
        'fno_times': fno_times,
        'fem_errors': fem_errors,
        'fno_errors': fno_errors
    }

def plot_comparison(source, fem_solution, fno_solution):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    im1 = axes[0].imshow(source, cmap='RdBu_r')
    axes[0].set_title('Source Term f(x,y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(fem_solution, cmap='viridis')
    axes[1].set_title('FEM Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(fno_solution, cmap='viridis')
    axes[2].set_title('FNO Solution')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    error = np.abs(fem_solution - fno_solution)
    im4 = axes[3].imshow(error, cmap='Reds')
    axes[3].set_title('Absolute Error')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    grid_size = 64
    n_train = 800
    n_val = 100
    n_test = 100
    batch_size = 32
    epochs = 100
    
    # Generate datasets
    print("Generating training data...")
    train_dataset = PoissonDataset(n_train, grid_size)
    
    print("\nGenerating validation data...")
    val_dataset = PoissonDataset(n_val, grid_size)
    
    print("\nGenerating test data...")
    test_dataset = PoissonDataset(n_test, grid_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train FNO
    print("\nTraining FNO model...")
    fno_model = FNO2d(modes1=16, modes2=16, width=64, n_layers=4)
    train_losses, val_losses = train_fno(fno_model, train_loader, val_loader, epochs=epochs)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('FNO Training Progress')
    plt.yscale('log')
    
    # Evaluate both methods
    print("\nEvaluating FEM vs FNO performance...")
    results = evaluate_methods(test_dataset, fno_model, grid_size)
    
    # Performance comparison
    avg_fem_time = np.mean(results['fem_times'])
    avg_fno_time = np.mean(results['fno_times'])
    avg_fem_error = np.mean(results['fem_errors'])
    avg_fno_error = np.mean(results['fno_errors'])
    
    print("\nPERFORMANCE COMPARISON")
    print(f"Average FEM time: {avg_fem_time:.4f} seconds")
    print(f"Average FNO time: {avg_fno_time:.4f} seconds")
    print(f"Speed-up factor: {avg_fem_time/avg_fno_time:.2f}x")
    print(f"\nAverage FEM error (self-consistency): {avg_fem_error:.8f}")
    print(f"Average FNO error (vs FEM): {avg_fno_error:.8f}")
    print(f"Relative error: {avg_fno_error/avg_fem_error:.2f}")
    
    # Plot performance comparison
    plt.subplot(1, 2, 2)
    methods = ['FEM', 'FNO']
    times = [avg_fem_time, avg_fno_time]
    plt.bar(methods, times, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Average Time (seconds)')
    plt.title('Computational Time Comparison')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    

    print("\nDisplaying example solutions...")
    test_idx = 0
    source, fem_solution = test_dataset[test_idx]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fno_model.eval()
    with torch.no_grad():
        source_tensor = source.unsqueeze(0).to(device)
        fno_solution = fno_model(source_tensor).cpu().numpy()[0]
    
    plot_comparison(source.numpy(), fem_solution.numpy(), fno_solution)

if __name__ == "__main__":
    main()