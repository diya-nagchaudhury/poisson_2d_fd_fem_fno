"""
Finite Difference solver for 2D Poisson equation
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time
from typing import Tuple, List

from common_functions import (
    rhs_function, exact_solution, create_coordinate_grid,
    compute_l2_error, compute_h1_error, save_solution
)
from plotting_utils import plot_solution_comparison, plot_convergence_study


def create_2d_laplacian(nx: int, ny: int, dx: float, dy: float) -> csr_matrix:
    """
    Create 2D Laplacian matrix using finite differences
    5-point stencil: [0, 1, -4, 1, 0] in x and y directions
    
    Args:
        nx, ny: Number of grid points in x and y directions
        dx, dy: Grid spacing in x and y directions
        
    Returns:
        A: Sparse Laplacian matrix
    """
    N = nx * ny
    
    # Coefficients for the 5-point stencil
    cx = 1.0 / (dx**2)
    cy = 1.0 / (dy**2)
    cc = -2.0 * (cx + cy)
    
    # Main diagonal
    main_diag = cc * np.ones(N)
    
    # Off-diagonals for x-direction (±1 positions)
    x_diag = cx * np.ones(N-1)
    # Remove connections across y-boundaries
    for i in range(nx-1, N-1, nx):
        x_diag[i] = 0
    
    # Off-diagonals for y-direction (±nx positions)
    y_diag = cy * np.ones(N-nx)
    
    # Create sparse matrix
    A = diags([y_diag, x_diag, main_diag, x_diag, y_diag], 
              [-nx, -1, 0, 1, nx], 
              shape=(N, N), format='csr')
    
    return A


def apply_boundary_conditions(A: csr_matrix, b: np.ndarray, nx: int, ny: int) -> Tuple[csr_matrix, np.ndarray, List[int]]:
    """
    Apply homogeneous Dirichlet boundary conditions u = 0
    
    Args:
        A: System matrix
        b: Right-hand side vector
        nx, ny: Grid dimensions
        
    Returns:
        A: Modified system matrix
        b: Modified right-hand side vector
        boundary_indices: List of boundary node indices
    """
    N = nx * ny
    
    # Boundary indices
    boundary_indices = []
    
    # Bottom and top boundaries
    boundary_indices.extend(range(nx))  # Bottom row
    boundary_indices.extend(range(N-nx, N))  # Top row
    
    # Left and right boundaries
    for i in range(nx, N-nx, nx):
        boundary_indices.append(i)  # Left column
        boundary_indices.append(i + nx - 1)  # Right column
    
    # Remove duplicates and sort
    boundary_indices = sorted(list(set(boundary_indices)))
    
    # Apply boundary conditions
    for idx in boundary_indices:
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
    
    return A, b, boundary_indices


class PoissonFDSolver:
    """
    Finite Difference solver for 2D Poisson equation
    """
    
    def __init__(self, nx: int = 33, ny: int = 33):
        """
        Initialize the FD solver
        
        Args:
            nx, ny: Number of grid points in x and y directions (including boundaries)
        """
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # Create grid
        self.x, self.y, self.X, self.Y = create_coordinate_grid(nx, ny)
        
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Solve the 2D Poisson equation using finite differences
        
        Returns:
            u: Solution array
            u_exact: Exact solution
            X, Y: Grid coordinates
            error_L2, error_H1: Errors compared to exact solution
        """
        
        # Right-hand side
        f = rhs_function(self.X, self.Y)
        
        # Flatten for linear system
        f_flat = f.flatten()
        
        # Create Laplacian matrix
        A = create_2d_laplacian(self.nx, self.ny, self.dx, self.dy)
        
        # Apply boundary conditions
        A, f_flat, boundary_indices = apply_boundary_conditions(A, f_flat, self.nx, self.ny)
        
        # Solve linear system
        print(f"Solving linear system of size {self.nx*self.ny} x {self.nx*self.ny}...")
        start_time = time.time()
        u_flat = spsolve(A, f_flat)
        solve_time = time.time() - start_time
        print(f"Linear system solved in {solve_time:.3f} seconds")
        
        # Reshape solution
        u = u_flat.reshape((self.nx, self.ny))
        
        # Compute exact solution
        u_exact = exact_solution(self.X, self.Y)
        
        # Compute errors
        error_L2 = compute_l2_error(u, u_exact, self.dx, self.dy)
        error_H1 = compute_h1_error(u, u_exact, self.dx, self.dy)
        
        return u, u_exact, self.X, self.Y, error_L2, error_H1


def convergence_study_fd(grid_sizes: List[int] = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Perform convergence study with different grid sizes
    
    Args:
        grid_sizes: List of grid sizes to test
        
    Returns:
        h_values: Grid spacing values
        L2_errors: L2 errors
        H1_errors: H1 errors
    """
    if grid_sizes is None:
        grid_sizes = [17, 33, 65, 129]  # nx = ny (odd numbers for centered differences)
    
    h_values = []
    L2_errors = []
    H1_errors = []
    
    print("Finite Difference Convergence Study:")
    print("=" * 70)
    print(f"{'Grid Size':<10} {'h':<12} {'DOFs':<10} {'L2 Error':<15} {'H1 Error':<15}")
    print("-" * 70)
    
    for nx in grid_sizes:
        # Solve problem
        solver = PoissonFDSolver(nx, nx)
        u, u_exact, X, Y, L2_error, H1_error = solver.solve()
        
        # Grid size parameter
        h = 1.0 / (nx - 1)
        dofs = nx * nx
        
        h_values.append(h)
        L2_errors.append(L2_error)
        H1_errors.append(H1_error)
        
        print(f"{nx:<10} {h:<12.4f} {dofs:<10} {L2_error:<15.6e} {H1_error:<15.6e}")
    
    # Compute convergence rates
    if len(L2_errors) > 1:
        print("\nConvergence Rates:")
        print("-" * 50)
        for i in range(1, len(L2_errors)):
            rate_L2 = np.log(L2_errors[i-1]/L2_errors[i]) / np.log(h_values[i-1]/h_values[i])
            rate_H1 = np.log(H1_errors[i-1]/H1_errors[i]) / np.log(h_values[i-1]/h_values[i])
            print(f"h={h_values[i]:.4f}: L2 rate = {rate_L2:.2f}, H1 rate = {rate_H1:.2f}")
    
    return h_values, L2_errors, H1_errors


def main():
    """
    Main function to run the finite difference Poisson solver
    """
    print("2D Poisson Equation Solver using Finite Differences")
    print("==================================================")
    print("Solving: -∇²u = f in Ω = [0,1]²")
    print("with u = 0 on ∂Ω")
    print("Using 5-point stencil finite difference scheme")
    print()
    
    # Solve with default parameters
    print("Solving with 65x65 grid...")
    solver = PoissonFDSolver(nx=65, ny=65)
    u, u_exact, X, Y, L2_error, H1_error = solver.solve()
    
    print(f"L2 Error: {L2_error:.6e}")
    print(f"H1 Error: {H1_error:.6e}")
    print()
    
    # Perform convergence study
    h_vals, L2_errs, H1_errs = convergence_study_fd()
    
    # Plot solution
    plot_solution_comparison(u, X, Y, u_exact, "FD", "poisson_fd_solution_comparison")
    
    # Plot convergence
    plot_convergence_study(h_vals, L2_errs, H1_errs, "Finite Difference Convergence Study")
    
    # Save solution
    save_solution(u, X, Y, "poisson_fd_solution")
    
    return u, X, Y, u_exact


if __name__ == "__main__":
    # Run the solver
    u, X, Y, u_exact = main()