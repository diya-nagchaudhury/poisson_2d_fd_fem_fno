"""
Fourier Neural Operator (FNO) model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SpectralConv2d(nn.Module):
    """
    2D Spectral convolution layer for FNO
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initialize spectral convolution layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication for 2D tensors
        
        Args:
            input: Input tensor (batch, in_channel, x, y)
            weights: Weight tensor (in_channel, out_channel, x, y)
            
        Returns:
            Output tensor (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spectral convolution
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after spectral convolution
        """
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator model for solving PDEs
    """
    
    def __init__(self, modes1: int, modes2: int, width: int, layers: int = 4):
        """
        Initialize FNO2d model
        
        Args:
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
            width: Width of the hidden layers
            layers: Number of FNO layers
        """
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.layers = layers
        
        # Input projection
        self.fc0 = nn.Linear(3, self.width)  # input: (x, y, rhs)
        
        # FNO layers
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2) 
            for _ in range(self.layers)
        ])
        
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) 
            for _ in range(self.layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FNO2d
        
        Args:
            x: Input tensor (batch, x_dim, y_dim, 3) where 3 = (x_coord, y_coord, rhs)
            
        Returns:
            Output tensor (batch, x_dim, y_dim)
        """
        # Input projection
        x = self.fc0(x)  # (batch, x_dim, y_dim, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, x_dim, y_dim)
        
        # FNO layers
        for i in range(self.layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.layers - 1:
                x = F.gelu(x)
        
        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, x_dim, y_dim, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (batch, x_dim, y_dim, 1)
        
        return x.squeeze(-1)  # (batch, x_dim, y_dim)


def get_default_fno_config() -> Dict:
    """
    Get default configuration for FNO model
    
    Returns:
        Default configuration dictionary
    """
    return {
        'modes1': 12,           # Number of Fourier modes in x direction
        'modes2': 12,           # Number of Fourier modes in y direction
        'width': 32,            # Width of the FNO layers
        'layers': 4,            # Number of FNO layers
        'epochs': 1000,
        'batch_size': 10,
        'learning_rate': {
            'initial_learning_rate': 0.001,
            'use_lr_scheduler': True,
            'decay_steps': 200,
            'decay_rate': 0.95
        },
        'geometry': {
            'x_min': 0, 'x_max': 1,
            'y_min': 0, 'y_max': 1,
            'nx': 64,   # Grid resolution
            'ny': 64
        }
    }


def relative_l2_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute relative L2 error between prediction and true values
    
    Args:
        pred: Predicted values
        true: True values
        
    Returns:
        Relative L2 error
    """
    return torch.norm(pred - true) / torch.norm(true)