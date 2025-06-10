import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fno_model import FNO2d, get_default_fno_config, relative_l2_error
from common_functions import *

def create_training_data(n_samples=1000, nx=64, ny=64):
    """Create training data for FNO"""
    # Create coordinate grid
    x, y, X, Y = create_coordinate_grid(nx, ny)
    
    # Create input features (x, y coordinates + RHS)
    inputs = []
    outputs = []
    
    for i in range(n_samples):
        # You can vary the RHS function parameters here for different samples
        # For now, using the same RHS
        rhs = rhs_function(X, Y)
        u_exact = exact_solution(X, Y)
        
        # Input: (x_coord, y_coord, rhs)
        input_sample = np.stack([X, Y, rhs], axis=-1)
        inputs.append(input_sample)
        outputs.append(u_exact)
    
    return np.array(inputs), np.array(outputs)

def train_fno():
    """Train the FNO model"""
    config = get_default_fno_config()
    
    # Create model
    model = FNO2d(
        modes1=config['modes1'],
        modes2=config['modes2'], 
        width=config['width'],
        layers=config['layers']
    )
    
    # Create training data
    X_train, y_train = create_training_data(n_samples=100)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    
    # Create data loader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']['initial_learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.6e}")
    
    return model

# Train the model
model = train_fno()