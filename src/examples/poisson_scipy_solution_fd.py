import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time

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

def create_2d_laplacian(nx, ny, dx, dy):
    """
    Create 2D Laplacian matrix using finite differences
    5-point stencil: [0, 1, -4, 1, 0] in x and y directions
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

def apply_boundary_conditions(A, b, nx, ny):
    """
    Apply homogeneous Dirichlet boundary conditions u = 0
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

def solve_poisson_2d_fd(nx=33, ny=33):
    """
    Solve 2D Poisson equation using finite differences
    
    Parameters:
    nx, ny: Number of grid points in x and y directions (including boundaries)
    
    Returns:
    u: Solution array
    x, y: Grid coordinates
    error_L2, error_H1: Errors compared to exact solution
    """
    
    # Grid spacing
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # Create grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Right-hand side
    f = rhs_function(X, Y)
    
    # Flatten for linear system
    f_flat = f.flatten()
    
    # Create Laplacian matrix
    A = create_2d_laplacian(nx, ny, dx, dy)
    
    # Apply boundary conditions
    A, f_flat, boundary_indices = apply_boundary_conditions(A, f_flat, nx, ny)
    
    # Solve linear system
    print(f"Solving linear system of size {nx*ny} x {nx*ny}...")
    start_time = time.time()
    u_flat = spsolve(A, f_flat)
    solve_time = time.time() - start_time
    print(f"Linear system solved in {solve_time:.3f} seconds")
    
    # Reshape solution
    u = u_flat.reshape((nx, ny))
    
    # Compute exact solution
    u_exact = exact_solution(X, Y)
    
    # Compute errors
    error = u - u_exact
    error_L2 = np.sqrt(dx * dy * np.sum(error**2))
    
    # H1 error (approximate using finite differences)
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dudx_exact = np.zeros_like(u_exact)
    dudy_exact = np.zeros_like(u_exact)
    
    # Interior points for derivatives
    dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
    
    dudx_exact[1:-1, :] = (u_exact[2:, :] - u_exact[:-2, :]) / (2 * dx)
    dudy_exact[:, 1:-1] = (u_exact[:, 2:] - u_exact[:, :-2]) / (2 * dy)
    
    grad_error_x = dudx - dudx_exact
    grad_error_y = dudy - dudy_exact
    
    error_H1 = np.sqrt(dx * dy * np.sum(error**2 + grad_error_x**2 + grad_error_y**2))
    
    return u, X, Y, u_exact, error_L2, error_H1

def plot_solution_fd(u, X, Y, u_exact, title_prefix="Finite Difference"):
    """
    Plot the exact solution, numerical solution, and absolute error in a horizontal layout
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Compute absolute error
    abs_error = np.abs(u - u_exact)
    
    # True Solution (Exact)
    im1 = axes[0].contourf(X, Y, u_exact, levels=50, cmap='viridis')
    axes[0].set_title('True Solution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    
    # FD Prediction (Numerical)
    im2 = axes[1].contourf(X, Y, u, levels=50, cmap='viridis')
    axes[1].set_title('FD Prediction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    
    # Absolute Error
    im3 = axes[2].contourf(X, Y, abs_error, levels=50, cmap='Reds')
    axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    cbar3 = plt.colorbar(im3, ax=axes[2])
    
    # Make all subplots square
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    plt.savefig(f"{title_prefix}_solution_comparison.png", dpi=300, bbox_inches='tight')

def convergence_study_fd():
    """
    Perform convergence study with different grid sizes
    """
    grid_sizes = [17, 33, 65, 129]  # nx = ny (odd numbers for centered differences)
    h_values = []
    L2_errors = []
    H1_errors = []
    
    print("Finite Difference Convergence Study:")
    print("=" * 60)
    print(f"{'Grid Size':<10} {'h':<12} {'DOFs':<10} {'L2 Error':<15} {'H1 Error':<15}")
    print("-" * 60)
    
    for nx in grid_sizes:
        # Solve problem
        u, X, Y, u_exact, L2_error, H1_error = solve_poisson_2d_fd(nx, nx)
        
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
        print("-" * 40)
        for i in range(1, len(L2_errors)):
            rate_L2 = np.log(L2_errors[i-1]/L2_errors[i]) / np.log(h_values[i-1]/h_values[i])
            rate_H1 = np.log(H1_errors[i-1]/H1_errors[i]) / np.log(h_values[i-1]/h_values[i])
            print(f"h={h_values[i]:.4f}: L2 rate = {rate_L2:.2f}, H1 rate = {rate_H1:.2f}")
    
    return h_values, L2_errors, H1_errors

def save_solution_fd(u, X, Y, filename="poisson_fd_solution"):
    """
    Save solution to files
    """
    # Save as numpy arrays
    np.savez(f"{filename}.npz", u=u, X=X, Y=Y)
    
    # Save as CSV for external visualization
    data = np.column_stack([X.flatten(), Y.flatten(), u.flatten()])
    np.savetxt(f"{filename}.csv", data, delimiter=',', 
               header='x,y,u', comments='')
    
    print(f"Solution saved to {filename}.npz and {filename}.csv")

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
    u, X, Y, u_exact, L2_error, H1_error = solve_poisson_2d_fd(nx=65, ny=65)
    
    print(f"L2 Error: {L2_error:.6e}")
    print(f"H1 Error: {H1_error:.6e}")
    print()
    
    # Perform convergence study
    h_vals, L2_errs, H1_errs = convergence_study_fd()
    
    # Plot solution
    plot_solution_fd(u, X, Y, u_exact)
    
    # Save solution
    save_solution_fd(u, X, Y, "poisson_fd_solution")
    
    return u, X, Y, u_exact

if __name__ == "__main__":
    # Run the solver
    u, X, Y, u_exact = main()