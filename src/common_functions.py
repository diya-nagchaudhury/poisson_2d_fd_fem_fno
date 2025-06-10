"""
Common functions shared between FD and FNO implementations
for solving the 2D Poisson equation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def rhs_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side function for the Poisson equation
    
    Args:
        x, y: Coordinate arrays
        
    Returns:
        RHS values at given coordinates
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    f_temp = -2.0 * (omega_x**2) * (np.sin(omega_x * x) * np.sin(omega_y * y))
    return f_temp


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Exact solution for the Poisson equation
    
    Args:
        x, y: Coordinate arrays
        
    Returns:
        Exact solution values at given coordinates
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    return np.sin(omega_x * x) * np.sin(omega_y * y)


def create_coordinate_grid(nx: int, ny: int, 
                          x_min: float = 0.0, x_max: float = 1.0,
                          y_min: float = 0.0, y_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create coordinate grid for the domain
    
    Args:
        nx, ny: Number of grid points in x and y directions
        x_min, x_max: Domain bounds in x direction
        y_min, y_max: Domain bounds in y direction
        
    Returns:
        x, y: 1D coordinate arrays
        X, Y: 2D coordinate meshgrids
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return x, y, X, Y


def compute_l2_error(u_numerical: np.ndarray, u_exact: np.ndarray, 
                    dx: float, dy: float) -> float:
    """
    Compute L2 error between numerical and exact solutions
    
    Args:
        u_numerical: Numerical solution
        u_exact: Exact solution
        dx, dy: Grid spacing
        
    Returns:
        L2 error
    """
    error = u_numerical - u_exact
    return np.sqrt(dx * dy * np.sum(error**2))


def compute_h1_error(u_numerical: np.ndarray, u_exact: np.ndarray,
                    dx: float, dy: float) -> float:
    """
    Compute H1 error between numerical and exact solutions
    
    Args:
        u_numerical: Numerical solution
        u_exact: Exact solution
        dx, dy: Grid spacing
        
    Returns:
        H1 error
    """
    error = u_numerical - u_exact
    
    # Compute derivatives using finite differences
    dudx = np.zeros_like(u_numerical)
    dudy = np.zeros_like(u_numerical)
    dudx_exact = np.zeros_like(u_exact)
    dudy_exact = np.zeros_like(u_exact)
    
    # Interior points for derivatives
    dudx[1:-1, :] = (u_numerical[2:, :] - u_numerical[:-2, :]) / (2 * dx)
    dudy[:, 1:-1] = (u_numerical[:, 2:] - u_numerical[:, :-2]) / (2 * dy)
    
    dudx_exact[1:-1, :] = (u_exact[2:, :] - u_exact[:-2, :]) / (2 * dx)
    dudy_exact[:, 1:-1] = (u_exact[:, 2:] - u_exact[:, :-2]) / (2 * dy)
    
    grad_error_x = dudx - dudx_exact
    grad_error_y = dudy - dudy_exact
    
    return np.sqrt(dx * dy * np.sum(error**2 + grad_error_x**2 + grad_error_y**2))


def save_solution(u: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                 filename: str = "solution"):
    """
    Save solution to files
    
    Args:
        u: Solution array
        X, Y: Coordinate arrays
        filename: Base filename for output files
    """
    # Save as numpy arrays
    np.savez(f"{filename}.npz", u=u, X=X, Y=Y)
    
    # Save as CSV for external visualization
    data = np.column_stack([X.flatten(), Y.flatten(), u.flatten()])
    np.savetxt(f"{filename}.csv", data, delimiter=',', 
               header='x,y,u', comments='')
    
    print(f"Solution saved to {filename}.npz and {filename}.csv")