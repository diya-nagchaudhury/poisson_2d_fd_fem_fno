"""
Plotting utilities for visualizing solutions and errors
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_solution_comparison(u_numerical: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                           u_exact: np.ndarray, title_prefix: str = "Solution",
                           save_filename: Optional[str] = None):
    """
    Plot the exact solution, numerical solution, and absolute error in a horizontal layout
    
    Args:
        u_numerical: Numerical solution
        X, Y: Coordinate meshgrids
        u_exact: Exact solution
        title_prefix: Prefix for plot titles
        save_filename: Optional filename to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Compute absolute error
    abs_error = np.abs(u_numerical - u_exact)
    
    # True Solution (Exact)
    im1 = axes[0].contourf(X, Y, u_exact, levels=50, cmap='viridis')
    axes[0].set_title('True Solution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    
    # Numerical Prediction
    im2 = axes[1].contourf(X, Y, u_numerical, levels=50, cmap='viridis')
    axes[1].set_title(f'{title_prefix} Prediction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    
    # Absolute Error
    im3 = axes[2].contourf(X, Y, abs_error, levels=50, cmap='Reds')
    axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    cbar3 = plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure if filename provided
    if save_filename:
        plt.savefig(f"{save_filename}.png", dpi=300, bbox_inches='tight')


def plot_convergence_study(h_values: List[float], L2_errors: List[float], 
                          H1_errors: List[float], title: str = "Convergence Study"):
    """
    Plot convergence study results
    
    Args:
        h_values: Grid spacing values
        L2_errors: L2 errors
        H1_errors: H1 errors
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 error plot
    ax1.loglog(h_values, L2_errors, 'o-', label='L2 Error', linewidth=2, markersize=8)
    ax1.loglog(h_values, [h**2 for h in h_values], '--', label='O(h²)', alpha=0.7)
    ax1.set_xlabel('Grid spacing (h)')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('L2 Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # H1 error plot
    ax2.loglog(h_values, H1_errors, 's-', label='H1 Error', linewidth=2, markersize=8, color='red')
    ax2.loglog(h_values, h_values, '--', label='O(h)', alpha=0.7, color='red')
    ax2.set_xlabel('Grid spacing (h)')
    ax2.set_ylabel('H1 Error')
    ax2.set_title('H1 Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title.lower().replace(' ', '_')}_convergence.png", dpi=300, bbox_inches='tight')


def plot_training_history(train_losses: List[float], val_losses: List[float] = None,
                         title: str = "Training History"):
    """
    Plot training history for neural network models
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Full training history
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title('Full Training History')
    axes[0].grid(True, alpha=0.3)
    
    # Last portion of training
    last_n = min(100, len(train_losses))
    axes[1].plot(train_losses[-last_n:], label=f'Train Loss (last {last_n})', linewidth=2)
    if val_losses:
        axes[1].plot(val_losses[-last_n:], label=f'Val Loss (last {last_n})', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].set_title(f'Training History (Final {last_n} Epochs)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')