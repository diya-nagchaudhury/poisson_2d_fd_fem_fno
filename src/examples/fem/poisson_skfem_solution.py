import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import *
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

@BilinearForm
def laplacian(u, v, _):
    """
    Bilinear form for the Laplacian: ∫∇u·∇v dx
    """
    return ddot(dd(u), dd(v))

@LinearForm  
def load(v, w):
    """
    Linear form for the right-hand side: ∫f·v dx
    """
    x, y = w.x
    f_vals = rhs_function(x, y)
    return f_vals * v

def solve_poisson_2d_skfem(n_refine=5, element_type='P1'):
    """
    Solve 2D Poisson equation using scikit-fem
    
    Parameters:
    n_refine: Number of mesh refinements (mesh size ≈ 2^(-n_refine))
    element_type: 'P1' or 'P2' for linear or quadratic elements
    
    Returns:
    u: Solution array
    m: Mesh object
    basis: Basis object
    error_L2, error_H1: Errors compared to exact solution
    """
    
    # Create mesh - unit square (FIXED)
    # Use the correct way to create and refine mesh in recent scikit-fem versions
    m = MeshTri.unit_square()
    for _ in range(n_refine):
        m = m.refined()
    
    # Create basis (function space)
    if element_type == 'P1':
        basis = CellBasis(m, ElementTriP1())
    elif element_type == 'P2':
        basis = CellBasis(m, ElementTriP2()) 
    else:
        raise ValueError("element_type must be 'P1' or 'P2'")
    
    print(f"Mesh info: {m.p.shape[1]} vertices, {m.t.shape[1]} triangles")
    print(f"DOFs: {basis.N}")
    
    # Assemble system matrices
    print("Assembling system matrices...")
    start_time = time.time()
    
    A = asm(laplacian, basis)
    b = asm(load, basis)
    
    assembly_time = time.time() - start_time
    print(f"Assembly completed in {assembly_time:.3f} seconds")
    
    # Apply boundary conditions (homogeneous Dirichlet)
    boundary_nodes = m.boundary()
    u = solve(*condense(A, b, I=boundary_nodes))
    
    solve_time = time.time() - start_time - assembly_time
    print(f"Linear system solved in {solve_time:.3f} seconds")
    
    # Compute exact solution at mesh nodes
    x, y = m.p[0, :], m.p[1, :]
    u_exact = exact_solution(x, y)
    
    # Compute errors
    error_L2 = np.sqrt(asm(lambda u, v, w: (u - v)**2, basis, u, u_exact))
    
    # H1 error (including gradient terms)
    @BilinearForm
    def h1_error_form(u_err, v_err, _):
        return u_err * v_err + ddot(dd(u_err), dd(v_err))
    
    u_error = u - u_exact
    error_H1 = np.sqrt(asm(h1_error_form, basis, u_error, u_error))
    
    return u, m, basis, u_exact, error_L2, error_H1

def convergence_study_skfem():
    """
    Perform convergence study with different mesh refinements
    """
    refinement_levels = [3, 4, 5, 6]
    h_values = []
    L2_errors = []
    H1_errors = []
    dofs_list = []
    
    print("Scikit-fem Convergence Study:")
    print("=" * 65)
    print(f"{'Refine Level':<12} {'h':<12} {'DOFs':<10} {'L2 Error':<15} {'H1 Error':<15}")
    print("-" * 65)
    
    for n_refine in refinement_levels:
        # Solve problem
        u, m, basis, u_exact, L2_error, H1_error = solve_poisson_2d_skfem(n_refine)
        
        # Estimate mesh size
        h = 2**(-n_refine)
        dofs = basis.N
        
        h_values.append(h)
        L2_errors.append(L2_error)
        H1_errors.append(H1_error)
        dofs_list.append(dofs)
        
        print(f"{n_refine:<12} {h:<12.4f} {dofs:<10} {L2_error:<15.6e} {H1_error:<15.6e}")
    
    # Compute convergence rates
    if len(L2_errors) > 1:
        print("\nConvergence Rates:")
        print("-" * 40)
        for i in range(1, len(L2_errors)):
            rate_L2 = np.log(L2_errors[i-1]/L2_errors[i]) / np.log(h_values[i-1]/h_values[i])
            rate_H1 = np.log(H1_errors[i-1]/H1_errors[i]) / np.log(h_values[i-1]/h_values[i])
            print(f"h={h_values[i]:.4f}: L2 rate = {rate_L2:.2f}, H1 rate = {rate_H1:.2f}")
    
    return h_values, L2_errors, H1_errors, dofs_list

def plot_solution_skfem(u, m, u_exact, title_prefix="Scikit-fem"):
    """
    Plot the numerical solution, exact solution, and error
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Solution Results', fontsize=16)
    
    # Numerical solution
    m.plot(u, ax=axes[0,0], shading='gouraud', colorbar=True)
    axes[0,0].set_title('Numerical Solution')
    axes[0,0].set_aspect('equal')
    
    # Exact solution  
    m.plot(u_exact, ax=axes[0,1], shading='gouraud', colorbar=True)
    axes[0,1].set_title('Exact Solution')
    axes[0,1].set_aspect('equal')
    
    # Error
    error = u - u_exact
    m.plot(error, ax=axes[1,0], shading='gouraud', colorbar=True)
    axes[1,0].set_title('Error (Numerical - Exact)')
    axes[1,0].set_aspect('equal')
    
    # Mesh visualization
    m.plot(ax=axes[1,1], boundaries_only=False)
    axes[1,1].set_title(f'Mesh ({m.t.shape[1]} triangles)')
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def plot_convergence(h_values, L2_errors, H1_errors):
    """
    Plot convergence rates
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 error
    ax1.loglog(h_values, L2_errors, 'bo-', label='L2 Error', linewidth=2, markersize=8)
    ax1.loglog(h_values, [h**2 for h in h_values], 'r--', label='h²', alpha=0.7)
    ax1.set_xlabel('Mesh size h')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('L2 Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # H1 error
    ax2.loglog(h_values, H1_errors, 'go-', label='H1 Error', linewidth=2, markersize=8)
    ax2.loglog(h_values, h_values, 'r--', label='h', alpha=0.7)
    ax2.set_xlabel('Mesh size h')
    ax2.set_ylabel('H1 Error')
    ax2.set_title('H1 Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_solution_skfem(u, m, u_exact, filename="poisson_skfem_solution"):
    """
    Save solution and mesh to files
    """
    # Save solution data
    x, y = m.p[0, :], m.p[1, :]
    data = np.column_stack([x, y, u, u_exact])
    np.savetxt(f"{filename}.csv", data, delimiter=',', 
               header='x,y,u_numerical,u_exact', comments='')
    
    # Save mesh connectivity
    np.savetxt(f"{filename}_mesh.csv", m.t.T, delimiter=',', fmt='%d',
               header='node1,node2,node3', comments='')
    
    print(f"Solution saved to {filename}.csv")
    print(f"Mesh connectivity saved to {filename}_mesh.csv")

def compare_element_types():
    """
    Compare P1 vs P2 elements
    """
    print("\nElement Type Comparison:")
    print("=" * 50)
    
    n_refine = 4
    
    for element_type in ['P1', 'P2']:
        print(f"\n{element_type} Elements:")
        print("-" * 20)
        u, m, basis, u_exact, L2_error, H1_error = solve_poisson_2d_skfem(
            n_refine, element_type
        )
        print(f"L2 Error: {L2_error:.6e}")
        print(f"H1 Error: {H1_error:.6e}")
        print(f"DOFs: {basis.N}")

def main():
    """
    Main function to run the scikit-fem Poisson solver
    """
    print("2D Poisson Equation Solver using scikit-fem")
    print("===========================================")
    print("Solving: -∇²u = f in Ω = [0,1]²")
    print("with u = 0 on ∂Ω")
    print("Using linear triangular finite elements")
    print()
    
    # Check if scikit-fem is installed
    try:
        import skfem
        print(f"scikit-fem version: {skfem.__version__}")
    except ImportError:
        print("scikit-fem not found. Install with: pip install scikit-fem")
        return None, None, None, None
    
    # Solve with default parameters
    print("\nSolving with refinement level 5...")
    u, m, basis, u_exact, L2_error, H1_error = solve_poisson_2d_skfem(n_refine=5)
    
    print(f"L2 Error: {L2_error:.6e}")
    print(f"H1 Error: {H1_error:.6e}")
    print()
    
    # Perform convergence study
    h_vals, L2_errs, H1_errs, dofs_list = convergence_study_skfem()
    
    # Compare element types
    compare_element_types()
    
    # Plot solution
    plot_solution_skfem(u, m, u_exact)
    
    # Plot convergence
    plot_convergence(h_vals, L2_errs, H1_errs)
    
    # Save solution
    save_solution_skfem(u, m, u_exact, "poisson_skfem_solution")
    
    return u, m, basis, u_exact

if __name__ == "__main__":
    # Run the solver
    u, m, basis, u_exact = main()