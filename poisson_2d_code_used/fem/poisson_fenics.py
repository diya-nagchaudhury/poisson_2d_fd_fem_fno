import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

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

class RHSExpression(UserExpression):
    """FEniCS expression for the right-hand side function"""
    def eval(self, value, x):
        value[0] = rhs_function(x[0], x[1])
    
    def value_shape(self):
        return ()

class ExactSolutionExpression(UserExpression):
    """FEniCS expression for the exact solution"""
    def eval(self, value, x):
        value[0] = exact_solution(x[0], x[1])
    
    def value_shape(self):
        return ()

def solve_poisson_2d(nx=32, ny=32, degree=1):
    """
    Solve 2D Poisson equation using FEniCS
    
    Parameters:
    nx, ny: Number of elements in x and y directions
    degree: Polynomial degree for finite elements
    
    Returns:
    u: Solution function
    mesh: The computational mesh
    V: Function space
    """
    
    # Create mesh - unit square [0,1] x [0,1]
    mesh = UnitSquareMesh(nx, ny)
    
    # Define function space
    V = FunctionSpace(mesh, 'P', degree)
    
    # Define boundary condition (homogeneous Dirichlet)
    def boundary(x, on_boundary):
        return on_boundary
    
    bc = DirichletBC(V, Constant(0.0), boundary)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Right-hand side function
    f = RHSExpression(degree=2)
    
    # Bilinear and linear forms
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Solve the system
    u = Function(V)
    solve(a == L, u, bc)
    
    return u, mesh, V

def compute_error(u_numerical, mesh, V):
    """
    Compute L2 and H1 errors compared to exact solution
    """
    # Exact solution
    u_exact = ExactSolutionExpression(degree=2)
    
    # L2 error
    error_L2 = errornorm(u_exact, u_numerical, 'L2')
    
    # H1 error
    error_H1 = errornorm(u_exact, u_numerical, 'H1')
    
    return error_L2, error_H1

def convergence_study():
    """
    Perform convergence study with different mesh sizes
    """
    mesh_sizes = [8, 16, 32, 64]
    h_values = []
    L2_errors = []
    H1_errors = []
    
    print("Convergence Study:")
    print("=" * 50)
    print(f"{'Mesh Size':<10} {'h':<12} {'L2 Error':<15} {'H1 Error':<15}")
    print("-" * 50)
    
    for nx in mesh_sizes:
        # Solve problem
        u, mesh, V = solve_poisson_2d(nx, nx, degree=1)
        
        # Compute errors
        L2_error, H1_error = compute_error(u, mesh, V)
        
        # Mesh size parameter
        h = 1.0 / nx
        
        h_values.append(h)
        L2_errors.append(L2_error)
        H1_errors.append(H1_error)
        
        print(f"{nx:<10} {h:<12.4f} {L2_error:<15.6e} {H1_error:<15.6e}")
    
    # Compute convergence rates
    if len(L2_errors) > 1:
        print("\nConvergence Rates:")
        print("-" * 30)
        for i in range(1, len(L2_errors)):
            rate_L2 = np.log(L2_errors[i-1]/L2_errors[i]) / np.log(h_values[i-1]/h_values[i])
            rate_H1 = np.log(H1_errors[i-1]/H1_errors[i]) / np.log(h_values[i-1]/h_values[i])
            print(f"h={h_values[i]:.4f}: L2 rate = {rate_L2:.2f}, H1 rate = {rate_H1:.2f}")
    
    return h_values, L2_errors, H1_errors

def plot_solution(u, mesh, title="FEM Solution"):
    """
    Plot the numerical solution
    """
    plt.figure(figsize=(12, 5))
    
    # Plot numerical solution
    plt.subplot(1, 2, 1)
    p1 = plot(u, title="Numerical Solution")
    plt.colorbar(p1)
    
    # Plot exact solution for comparison
    plt.subplot(1, 2, 2)
    V = u.function_space()
    u_exact = ExactSolutionExpression(degree=2)
    u_exact_proj = project(u_exact, V)
    p2 = plot(u_exact_proj, title="Exact Solution")
    plt.colorbar(p2)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the Poisson solver
    """
    print("2D Poisson Equation Solver using FEniCS")
    print("=======================================")
    print("Solving: -∇²u = f in Ω = [0,1]²")
    print("with u = 0 on ∂Ω")
    print()
    
    # Solve with default parameters
    print("Solving with 32x32 mesh...")
    u, mesh, V = solve_poisson_2d(nx=32, ny=32, degree=1)
    
    # Compute and display errors
    L2_error, H1_error = compute_error(u, mesh, V)
    print(f"L2 Error: {L2_error:.6e}")
    print(f"H1 Error: {H1_error:.6e}")
    print()
    
    # Perform convergence study
    h_vals, L2_errs, H1_errs = convergence_study()
    
    # Plot solution (uncomment if running interactively)
    # plot_solution(u, mesh)
    
    # Save solution to file
    file = File("poisson_solution.pvd")
    file << u
    print("\nSolution saved to 'poisson_solution.pvd'")
    
    return u, mesh, V

if __name__ == "__main__":
    # Set FEniCS parameters for better output
    set_log_level(LogLevel.ERROR)  # Reduce output verbosity
    
    # Run the solver
    u, mesh, V = main()