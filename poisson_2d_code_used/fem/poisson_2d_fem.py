# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix, lil_matrix
# from scipy.sparse.linalg import spsolve
# import time
# from typing import Tuple, List
# import matplotlib.tri as tri

# class QuadrilateralMesh:
#     """
#     Generate a structured quadrilateral mesh for the unit square
#     """
#     def __init__(self, nx: int, ny: int, x_min: float = 0, x_max: float = 1,
#                  y_min: float = 0, y_max: float = 1):
#         self.nx = nx  # Number of elements in x direction
#         self.ny = ny  # Number of elements in y direction
#         self.x_min, self.x_max = x_min, x_max
#         self.y_min, self.y_max = y_min, y_max
        
#         # Generate nodes
#         self.nodes = self._generate_nodes()
#         self.elements = self._generate_elements()
#         self.n_nodes = len(self.nodes)
#         self.n_elements = len(self.elements)
        
#         # Boundary information
#         self.boundary_nodes = self._get_boundary_nodes()
        
#     def _generate_nodes(self) -> np.ndarray:
#         """Generate node coordinates"""
#         x = np.linspace(self.x_min, self.x_max, self.nx + 1)
#         y = np.linspace(self.y_min, self.y_max, self.ny + 1)
        
#         nodes = []
#         for j in range(self.ny + 1):
#             for i in range(self.nx + 1):
#                 nodes.append([x[i], y[j]])
        
#         return np.array(nodes)
    
#     def _generate_elements(self) -> np.ndarray:
#         """Generate element connectivity (quad elements)"""
#         elements = []
#         for j in range(self.ny):
#             for i in range(self.nx):
#                 # Node indices for quadrilateral element
#                 n1 = j * (self.nx + 1) + i
#                 n2 = j * (self.nx + 1) + i + 1
#                 n3 = (j + 1) * (self.nx + 1) + i + 1
#                 n4 = (j + 1) * (self.nx + 1) + i
#                 elements.append([n1, n2, n3, n4])
        
#         return np.array(elements)
    
#     def _get_boundary_nodes(self) -> List[int]:
#         """Get indices of boundary nodes"""
#         boundary = []
        
#         # Bottom boundary (j=0)
#         for i in range(self.nx + 1):
#             boundary.append(i)
        
#         # Right boundary (i=nx)
#         for j in range(1, self.ny + 1):
#             boundary.append(j * (self.nx + 1) + self.nx)
        
#         # Top boundary (j=ny)
#         for i in range(self.nx - 1, -1, -1):
#             boundary.append(self.ny * (self.nx + 1) + i)
        
#         # Left boundary (i=0)
#         for j in range(self.ny - 1, 0, -1):
#             boundary.append(j * (self.nx + 1))
        
#         return list(set(boundary))  # Remove duplicates

# class BilinearQuadElement:
#     """
#     Bilinear quadrilateral element for 2D problems
#     """
#     def __init__(self):
#         # Gauss quadrature points and weights for 2x2 integration
#         self.gauss_points = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
#                                      [1/np.sqrt(3), -1/np.sqrt(3)],
#                                      [1/np.sqrt(3), 1/np.sqrt(3)],
#                                      [-1/np.sqrt(3), 1/np.sqrt(3)]])
#         self.gauss_weights = np.array([1.0, 1.0, 1.0, 1.0])
    
#     def shape_functions(self, xi: float, eta: float) -> np.ndarray:
#         """Bilinear shape functions"""
#         N = np.array([
#             0.25 * (1 - xi) * (1 - eta),  # N1
#             0.25 * (1 + xi) * (1 - eta),  # N2
#             0.25 * (1 + xi) * (1 + eta),  # N3
#             0.25 * (1 - xi) * (1 + eta)   # N4
#         ])
#         return N
    
#     def shape_function_derivatives(self, xi: float, eta: float) -> np.ndarray:
#         """Derivatives of shape functions with respect to xi and eta"""
#         dN_dxi = np.array([
#             -0.25 * (1 - eta),  # dN1/dxi
#             0.25 * (1 - eta),   # dN2/dxi
#             0.25 * (1 + eta),   # dN3/dxi
#             -0.25 * (1 + eta)   # dN4/dxi
#         ])
        
#         dN_deta = np.array([
#             -0.25 * (1 - xi),   # dN1/deta
#             -0.25 * (1 + xi),   # dN2/deta
#             0.25 * (1 + xi),    # dN3/deta
#             0.25 * (1 - xi)     # dN4/deta
#         ])
        
#         return dN_dxi, dN_deta
    
#     def jacobian(self, nodes: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
#         """Compute Jacobian matrix and determinant"""
#         dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        
#         # Jacobian matrix
#         J = np.zeros((2, 2))
#         J[0, 0] = np.sum(dN_dxi * nodes[:, 0])   # dx/dxi
#         J[0, 1] = np.sum(dN_dxi * nodes[:, 1])   # dy/dxi
#         J[1, 0] = np.sum(dN_deta * nodes[:, 0])  # dx/deta
#         J[1, 1] = np.sum(dN_deta * nodes[:, 1])  # dy/deta
        
#         det_J = np.linalg.det(J)
        
#         return J, det_J
    
#     def compute_element_matrix(self, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Compute element stiffness matrix and load vector"""
#         K_elem = np.zeros((4, 4))
#         F_elem = np.zeros(4)
        
#         # Integrate using Gauss quadrature
#         for i, (xi, eta) in enumerate(self.gauss_points):
#             weight = self.gauss_weights[i]
            
#             # Shape functions and derivatives
#             N = self.shape_functions(xi, eta)
#             dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            
#             # Jacobian
#             J, det_J = self.jacobian(nodes, xi, eta)
#             J_inv = np.linalg.inv(J)
            
#             # Derivatives in physical coordinates
#             dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
#             dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
            
#             # Stiffness matrix contribution
#             for j in range(4):
#                 for k in range(4):
#                     K_elem[j, k] += (dN_dx[j] * dN_dx[k] + dN_dy[j] * dN_dy[k]) * det_J * weight
            
#             # Load vector contribution (RHS function)
#             x_gauss = np.sum(N * nodes[:, 0])
#             y_gauss = np.sum(N * nodes[:, 1])
#             f_val = rhs_function(x_gauss, y_gauss)
            
#             for j in range(4):
#                 F_elem[j] += N[j] * f_val * det_J * weight
        
#         return K_elem, F_elem

# def rhs_function(x: float, y: float) -> float:
#     """
#     Right-hand side function for the Poisson equation
#     """
#     omega_x = 4.0 * np.pi
#     omega_y = 4.0 * np.pi
#     return -2.0 * (omega_x**2) * (np.sin(omega_x * x) * np.sin(omega_y * y))

# def exact_solution(x: float, y: float) -> float:
#     """
#     Exact solution for the Poisson equation
#     """
#     omega_x = 4.0 * np.pi
#     omega_y = 4.0 * np.pi
#     return np.sin(omega_x * x) * np.sin(omega_y * y)

# class PoissonFEMSolver:
#     """
#     Finite Element Method solver for 2D Poisson equation
#     """
#     def __init__(self, mesh: QuadrilateralMesh):
#         self.mesh = mesh
#         self.element = BilinearQuadElement()
#         self.solution = None
        
#     def assemble_system(self) -> Tuple[csr_matrix, np.ndarray]:
#         """
#         Assemble global stiffness matrix and load vector
#         """
#         n_nodes = self.mesh.n_nodes
#         K_global = lil_matrix((n_nodes, n_nodes))
#         F_global = np.zeros(n_nodes)
        
#         print(f"Assembling system for {self.mesh.n_elements} elements...")
        
#         # Loop over elements
#         for elem_idx, element_nodes in enumerate(self.mesh.elements):
#             # Get element node coordinates
#             elem_coords = self.mesh.nodes[element_nodes]
            
#             # Compute element matrices
#             K_elem, F_elem = self.element.compute_element_matrix(elem_coords)
            
#             # Assemble into global system
#             for i in range(4):
#                 global_i = element_nodes[i]
#                 F_global[global_i] += F_elem[i]
                
#                 for j in range(4):
#                     global_j = element_nodes[j]
#                     K_global[global_i, global_j] += K_elem[i, j]
        
#         return K_global.tocsr(), F_global
    
#     def apply_boundary_conditions(self, K: csr_matrix, F: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
#         """
#         Apply homogeneous Dirichlet boundary conditions (u = 0 on boundary)
#         """
#         K_bc = K.tolil()
#         F_bc = F.copy()
        
#         # Apply boundary conditions
#         for node_idx in self.mesh.boundary_nodes:
#             # Set diagonal to 1 and row to zero
#             K_bc[node_idx, :] = 0
#             K_bc[node_idx, node_idx] = 1
#             F_bc[node_idx] = 0
        
#         return K_bc.tocsr(), F_bc
    
#     def solve(self) -> np.ndarray:
#         """
#         Solve the linear system
#         """
#         print("Assembling global system...")
#         start_time = time.time()
        
#         # Assemble system
#         K, F = self.assemble_system()
#         assembly_time = time.time() - start_time
#         print(f"Assembly completed in {assembly_time:.3f} seconds")
        
#         # Apply boundary conditions
#         K_bc, F_bc = self.apply_boundary_conditions(K, F)
        
#         # Solve linear system
#         print("Solving linear system...")
#         solve_start = time.time()
#         self.solution = spsolve(K_bc, F_bc)
#         solve_time = time.time() - solve_start
#         print(f"Linear solve completed in {solve_time:.3f} seconds")
        
#         return self.solution
    
#     def compute_error(self) -> Tuple[float, float]:
#         """
#         Compute L2 and H1 errors compared to exact solution
#         """
#         if self.solution is None:
#             raise ValueError("Solution not computed yet")
        
#         l2_error = 0.0
#         h1_error = 0.0
        
#         # Compute errors by integrating over elements
#         for elem_idx, element_nodes in enumerate(self.mesh.elements):
#             elem_coords = self.mesh.nodes[element_nodes]
#             elem_solution = self.solution[element_nodes]
            
#             # Integrate over element using Gauss quadrature
#             for i, (xi, eta) in enumerate(self.element.gauss_points):
#                 weight = self.element.gauss_weights[i]
                
#                 N = self.element.shape_functions(xi, eta)
#                 dN_dxi, dN_deta = self.element.shape_function_derivatives(xi, eta)
                
#                 # Physical coordinates
#                 x = np.sum(N * elem_coords[:, 0])
#                 y = np.sum(N * elem_coords[:, 1])
                
#                 # Jacobian
#                 J, det_J = self.element.jacobian(elem_coords, xi, eta)
#                 J_inv = np.linalg.inv(J)
                
#                 # FEM solution and its derivatives at integration point
#                 u_fem = np.sum(N * elem_solution)
                
#                 # Derivatives in physical coordinates
#                 dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
#                 dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
                
#                 du_dx_fem = np.sum(dN_dx * elem_solution)
#                 du_dy_fem = np.sum(dN_dy * elem_solution)
                
#                 # Exact solution and derivatives
#                 u_exact = exact_solution(x, y)
#                 omega_x = omega_y = 4.0 * np.pi
#                 du_dx_exact = omega_x * np.cos(omega_x * x) * np.sin(omega_y * y)
#                 du_dy_exact = omega_y * np.sin(omega_x * x) * np.cos(omega_y * y)
                
#                 # Error contributions
#                 error_u = u_fem - u_exact
#                 error_du_dx = du_dx_fem - du_dx_exact
#                 error_du_dy = du_dy_fem - du_dy_exact
                
#                 l2_error += error_u**2 * det_J * weight
#                 h1_error += (error_u**2 + error_du_dx**2 + error_du_dy**2) * det_J * weight
        
#         return np.sqrt(l2_error), np.sqrt(h1_error)

# def plot_fem_results(mesh: QuadrilateralMesh, solution: np.ndarray, title: str = "FEM Solution"):
#     """
#     Plot FEM solution using triangulation for smooth visualization
#     """
#     # Create triangular mesh for plotting
#     x = mesh.nodes[:, 0]
#     y = mesh.nodes[:, 1]
    
#     # Create triangulation from quad mesh
#     triangles = []
#     for quad in mesh.elements:
#         # Split each quad into two triangles
#         triangles.append([quad[0], quad[1], quad[2]])
#         triangles.append([quad[0], quad[2], quad[3]])
    
#     triangulation = tri.Triangulation(x, y, triangles)
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
#     # FEM solution
#     im1 = axes[0].tricontourf(triangulation, solution, levels=50, cmap='viridis')
#     axes[0].set_title(f'{title}')
#     axes[0].set_xlabel('x')
#     axes[0].set_ylabel('y')
#     axes[0].set_aspect('equal')
#     plt.colorbar(im1, ax=axes[0])
    
#     # Exact solution
#     exact_vals = np.array([exact_solution(node[0], node[1]) for node in mesh.nodes])
#     im2 = axes[1].tricontourf(triangulation, exact_vals, levels=50, cmap='viridis')
#     axes[1].set_title('Exact Solution')
#     axes[1].set_xlabel('x')
#     axes[1].set_ylabel('y')
#     axes[1].set_aspect('equal')
#     plt.colorbar(im2, ax=axes[1])
    
#     # Error
#     error = np.abs(solution - exact_vals)
#     im3 = axes[2].tricontourf(triangulation, error, levels=50, cmap='Reds')
#     axes[2].set_title('Absolute Error')
#     axes[2].set_xlabel('x')
#     axes[2].set_ylabel('y')
#     axes[2].set_aspect('equal')
#     plt.colorbar(im3, ax=axes[2])
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('fem_solution.png')

# def convergence_study():
#     """
#     Perform convergence study with different mesh sizes
#     """
#     mesh_sizes = [8, 16, 32, 64]
#     l2_errors = []
#     h1_errors = []
#     h_values = []
#     solve_times = []
    
#     print("=== FEM Convergence Study ===")
    
#     for nx in mesh_sizes:
#         print(f"\nSolving with {nx}x{nx} mesh...")
        
#         # Create mesh and solver
#         mesh = QuadrilateralMesh(nx, nx)
#         solver = PoissonFEMSolver(mesh)
        
#         # Solve
#         start_time = time.time()
#         solution = solver.solve()
#         solve_time = time.time() - start_time
        
#         # Compute errors
#         l2_error, h1_error = solver.compute_error()
#         h = 1.0 / nx  # mesh size
        
#         # Store results
#         l2_errors.append(l2_error)
#         h1_errors.append(h1_error)
#         h_values.append(h)
#         solve_times.append(solve_time)
        
#         print(f"  Nodes: {mesh.n_nodes}")
#         print(f"  Elements: {mesh.n_elements}")
#         print(f"  Solve time: {solve_time:.3f} seconds")
#         print(f"  L2 error: {l2_error:.6e}")
#         print(f"  H1 error: {h1_error:.6e}")
        
#         # Plot solution for finest mesh
#         if nx == mesh_sizes[-1]:
#             plot_fem_results(mesh, solution, f"FEM Solution ({nx}x{nx})")
    
#     # Plot convergence
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 3, 1)
#     plt.loglog(h_values, l2_errors, 'bo-', label='L2 error')
#     plt.loglog(h_values, h1_errors, 'ro-', label='H1 error')
#     plt.loglog(h_values, [h**2 for h in h_values], 'k--', label='h²')
#     plt.xlabel('Mesh size h')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.title('Convergence Study')
#     plt.grid(True)
    
#     plt.subplot(1, 3, 2)
#     plt.loglog([1/h for h in h_values], solve_times, 'go-')
#     plt.xlabel('1/h (mesh resolution)')
#     plt.ylabel('Solve time (seconds)')
#     plt.title('Computational Cost')
#     plt.grid(True)
    
#     plt.subplot(1, 3, 3)
#     dofs = [(nx+1)**2 for nx in mesh_sizes]
#     plt.loglog(dofs, l2_errors, 'bo-', label='L2 error')
#     plt.loglog(dofs, [1/np.sqrt(n) for n in dofs], 'k--', label='1/√n')
#     plt.xlabel('Degrees of Freedom')
#     plt.ylabel('L2 Error')
#     plt.legend()
#     plt.title('Error vs DOFs')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('convergence_study.png')
    
#     return mesh_sizes, l2_errors, h1_errors, solve_times

# def main():
#     """
#     Main function to demonstrate FEM solver
#     """
#     print("=== 2D Poisson Equation - Finite Element Method ===")
#     print("Solving: -∇²u = f(x,y) with u = 0 on boundary")
#     print("Domain: [0,1] × [0,1]")
    
#     # Single solve with medium resolution
#     print("\n=== Single Solution ===")
#     nx = ny = 32
#     mesh = QuadrilateralMesh(nx, ny)
#     solver = PoissonFEMSolver(mesh)
    
#     print(f"Mesh: {nx}×{ny} elements, {mesh.n_nodes} nodes")
    
#     # Solve
#     solution = solver.solve()
    
#     # Compute errors
#     l2_error, h1_error = solver.compute_error()
#     print(f"\nErrors:")
#     print(f"  L2 error: {l2_error:.6e}")
#     print(f"  H1 error: {h1_error:.6e}")
    
#     # Plot results
#     plot_fem_results(mesh, solution)
    
#     # Convergence study
#     print("\n" + "="*50)
#     convergence_study()
    
#     # # Comparison summary
#     # print("\n=== FEM vs FNO Comparison ===")
#     # print("FEM Characteristics:")
#     # print("  + Mathematically rigorous convergence guarantees")
#     # print("  + Works on complex geometries and unstructured meshes")
#     # print("  + Well-established theory and error analysis")
#     # print("  + Handles various boundary conditions naturally")
#     # print("  - Requires mesh generation")
#     # print("  - Solve time scales as O(n^1.5) for 2D problems")
#     # print("  - Must resolve each new problem separately")
    
#     # print("\nFNO Characteristics:")
#     # print("  + Very fast inference once trained")
#     # print("  + Resolution independent")
#     # print("  + Can handle families of similar problems")
#     # print("  + Natural for parametric studies")
#     # print("  - Requires training data")
#     # print("  - Limited to structured grids (in basic form)")
#     # print("  - Less theoretical guarantees")
#     # print("  - Training can be expensive")

# if __name__ == "__main__":
#     main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time
from typing import Tuple, List
import matplotlib.tri as tri

# Copy the original code with some modifications for analysis

class QuadrilateralMesh:
    """
    Generate a structured quadrilateral mesh for the unit square
    """
    def __init__(self, nx: int, ny: int, x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1):
        self.nx = nx  # Number of elements in x direction
        self.ny = ny  # Number of elements in y direction
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
        # Generate nodes
        self.nodes = self._generate_nodes()
        self.elements = self._generate_elements()
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        
        # Boundary information
        self.boundary_nodes = self._get_boundary_nodes()
        
    def _generate_nodes(self) -> np.ndarray:
        """Generate node coordinates"""
        x = np.linspace(self.x_min, self.x_max, self.nx + 1)
        y = np.linspace(self.y_min, self.y_max, self.ny + 1)
        
        nodes = []
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                nodes.append([x[i], y[j]])
        
        return np.array(nodes)
    
    def _generate_elements(self) -> np.ndarray:
        """Generate element connectivity (quad elements)"""
        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                # Node indices for quadrilateral element
                n1 = j * (self.nx + 1) + i
                n2 = j * (self.nx + 1) + i + 1
                n3 = (j + 1) * (self.nx + 1) + i + 1
                n4 = (j + 1) * (self.nx + 1) + i
                elements.append([n1, n2, n3, n4])
        
        return np.array(elements)
    
    def _get_boundary_nodes(self) -> List[int]:
        """Get indices of boundary nodes"""
        boundary = []
        
        # Bottom boundary (j=0)
        for i in range(self.nx + 1):
            boundary.append(i)
        
        # Right boundary (i=nx)
        for j in range(1, self.ny + 1):
            boundary.append(j * (self.nx + 1) + self.nx)
        
        # Top boundary (j=ny)
        for i in range(self.nx - 1, -1, -1):
            boundary.append(self.ny * (self.nx + 1) + i)
        
        # Left boundary (i=0)
        for j in range(self.ny - 1, 0, -1):
            boundary.append(j * (self.nx + 1))
        
        return list(set(boundary))  # Remove duplicates

class BilinearQuadElement:
    """
    Bilinear quadrilateral element for 2D problems
    """
    def __init__(self):
        # Gauss quadrature points and weights for 2x2 integration
        self.gauss_points = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                                     [1/np.sqrt(3), -1/np.sqrt(3)],
                                     [1/np.sqrt(3), 1/np.sqrt(3)],
                                     [-1/np.sqrt(3), 1/np.sqrt(3)]])
        self.gauss_weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Bilinear shape functions"""
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  # N1
            0.25 * (1 + xi) * (1 - eta),  # N2
            0.25 * (1 + xi) * (1 + eta),  # N3
            0.25 * (1 - xi) * (1 + eta)   # N4
        ])
        return N
    
    def shape_function_derivatives(self, xi: float, eta: float) -> np.ndarray:
        """Derivatives of shape functions with respect to xi and eta"""
        dN_dxi = np.array([
            -0.25 * (1 - eta),  # dN1/dxi
            0.25 * (1 - eta),   # dN2/dxi
            0.25 * (1 + eta),   # dN3/dxi
            -0.25 * (1 + eta)   # dN4/dxi
        ])
        
        dN_deta = np.array([
            -0.25 * (1 - xi),   # dN1/deta
            -0.25 * (1 + xi),   # dN2/deta
            0.25 * (1 + xi),    # dN3/deta
            0.25 * (1 - xi)     # dN4/deta
        ])
        
        return dN_dxi, dN_deta
    
    def jacobian(self, nodes: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
        """Compute Jacobian matrix and determinant"""
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        
        # Jacobian matrix
        J = np.zeros((2, 2))
        J[0, 0] = np.sum(dN_dxi * nodes[:, 0])   # dx/dxi
        J[0, 1] = np.sum(dN_dxi * nodes[:, 1])   # dy/dxi
        J[1, 0] = np.sum(dN_deta * nodes[:, 0])  # dx/deta
        J[1, 1] = np.sum(dN_deta * nodes[:, 1])  # dy/deta
        
        det_J = np.linalg.det(J)
        
        return J, det_J
    
    def compute_element_matrix(self, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute element stiffness matrix and load vector"""
        K_elem = np.zeros((4, 4))
        F_elem = np.zeros(4)
        
        # Integrate using Gauss quadrature
        for i, (xi, eta) in enumerate(self.gauss_points):
            weight = self.gauss_weights[i]
            
            # Shape functions and derivatives
            N = self.shape_functions(xi, eta)
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            
            # Jacobian
            J, det_J = self.jacobian(nodes, xi, eta)
            J_inv = np.linalg.inv(J)
            
            # Derivatives in physical coordinates
            dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
            dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
            
            # Stiffness matrix contribution
            for j in range(4):
                for k in range(4):
                    K_elem[j, k] += (dN_dx[j] * dN_dx[k] + dN_dy[j] * dN_dy[k]) * det_J * weight
            
            # Load vector contribution (RHS function)
            x_gauss = np.sum(N * nodes[:, 0])
            y_gauss = np.sum(N * nodes[:, 1])
            f_val = rhs_function(x_gauss, y_gauss)
            
            for j in range(4):
                F_elem[j] += N[j] * f_val * det_J * weight
        
        return K_elem, F_elem

def rhs_function(x: float, y: float) -> float:
    """
    Right-hand side function for the Poisson equation
    CORRECTED: Should be 2 * omega^2, not -2 * omega^2
    Since -∇²u = f, and ∇²(sin(ωx)sin(ωy)) = -2ω²sin(ωx)sin(ωy)
    Therefore f = -(-2ω²sin(ωx)sin(ωy)) = 2ω²sin(ωx)sin(ωy)
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    return 2.0 * (omega_x**2) * (np.sin(omega_x * x) * np.sin(omega_y * y))

def exact_solution(x: float, y: float) -> float:
    """
    Exact solution for the Poisson equation
    """
    omega_x = 4.0 * np.pi
    omega_y = 4.0 * np.pi
    return np.sin(omega_x * x) * np.sin(omega_y * y)

class PoissonFEMSolver:
    """
    Finite Element Method solver for 2D Poisson equation
    """
    def __init__(self, mesh: QuadrilateralMesh):
        self.mesh = mesh
        self.element = BilinearQuadElement()
        self.solution = None
        
    def assemble_system(self) -> Tuple[csr_matrix, np.ndarray]:
        """
        Assemble global stiffness matrix and load vector
        """
        n_nodes = self.mesh.n_nodes
        K_global = lil_matrix((n_nodes, n_nodes))
        F_global = np.zeros(n_nodes)
        
        print(f"Assembling system for {self.mesh.n_elements} elements...")
        
        # Loop over elements
        for elem_idx, element_nodes in enumerate(self.mesh.elements):
            # Get element node coordinates
            elem_coords = self.mesh.nodes[element_nodes]
            
            # Compute element matrices
            K_elem, F_elem = self.element.compute_element_matrix(elem_coords)
            
            # Assemble into global system
            for i in range(4):
                global_i = element_nodes[i]
                F_global[global_i] += F_elem[i]
                
                for j in range(4):
                    global_j = element_nodes[j]
                    K_global[global_i, global_j] += K_elem[i, j]
        
        return K_global.tocsr(), F_global
    
    def apply_boundary_conditions(self, K: csr_matrix, F: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply homogeneous Dirichlet boundary conditions (u = 0 on boundary)
        IMPROVED METHOD: Properly eliminate rows and columns
        """
        K_bc = K.tolil()
        F_bc = F.copy()
        
        # First, modify the equations for boundary nodes
        for node_idx in self.mesh.boundary_nodes:
            # Zero out the row
            K_bc[node_idx, :] = 0
            # Set diagonal to 1
            K_bc[node_idx, node_idx] = 1
            # Set RHS to 0 (homogeneous Dirichlet)
            F_bc[node_idx] = 0
            
        # Also need to zero out columns for boundary nodes to maintain symmetry
        # This is important for proper conditioning
        K_bc = K_bc.tocsr()
        K_bc = K_bc.tolil()
        
        for node_idx in self.mesh.boundary_nodes:
            # Zero out the column (except diagonal)
            K_bc[:, node_idx] = 0
            K_bc[node_idx, node_idx] = 1
        
        return K_bc.tocsr(), F_bc
    
    def solve(self) -> np.ndarray:
        """
        Solve the linear system
        """
        print("Assembling global system...")
        start_time = time.time()
        
        # Assemble system
        K, F = self.assemble_system()
        assembly_time = time.time() - start_time
        print(f"Assembly completed in {assembly_time:.3f} seconds")
        
        # Apply boundary conditions
        K_bc, F_bc = self.apply_boundary_conditions(K, F)
        
        # Solve linear system
        print("Solving linear system...")
        solve_start = time.time()
        self.solution = spsolve(K_bc, F_bc)
        solve_time = time.time() - solve_start
        print(f"Linear solve completed in {solve_time:.3f} seconds")
        
        return self.solution
    
    def compute_error(self) -> Tuple[float, float]:
        """
        Compute L2 and H1 errors compared to exact solution
        """
        if self.solution is None:
            raise ValueError("Solution not computed yet")
        
        l2_error = 0.0
        h1_error = 0.0
        
        # Compute errors by integrating over elements
        for elem_idx, element_nodes in enumerate(self.mesh.elements):
            elem_coords = self.mesh.nodes[element_nodes]
            elem_solution = self.solution[element_nodes]
            
            # Integrate over element using Gauss quadrature
            for i, (xi, eta) in enumerate(self.element.gauss_points):
                weight = self.element.gauss_weights[i]
                
                N = self.element.shape_functions(xi, eta)
                dN_dxi, dN_deta = self.element.shape_function_derivatives(xi, eta)
                
                # Physical coordinates
                x = np.sum(N * elem_coords[:, 0])
                y = np.sum(N * elem_coords[:, 1])
                
                # Jacobian
                J, det_J = self.element.jacobian(elem_coords, xi, eta)
                J_inv = np.linalg.inv(J)
                
                # FEM solution and its derivatives at integration point
                u_fem = np.sum(N * elem_solution)
                
                # Derivatives in physical coordinates
                dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
                dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
                
                du_dx_fem = np.sum(dN_dx * elem_solution)
                du_dy_fem = np.sum(dN_dy * elem_solution)
                
                # Exact solution and derivatives
                u_exact = exact_solution(x, y)
                omega_x = omega_y = 4.0 * np.pi
                du_dx_exact = omega_x * np.cos(omega_x * x) * np.sin(omega_y * y)
                du_dy_exact = omega_y * np.sin(omega_x * x) * np.cos(omega_y * y)
                
                # Error contributions
                error_u = u_fem - u_exact
                error_du_dx = du_dx_fem - du_dx_exact
                error_du_dy = du_dy_fem - du_dy_exact
                
                l2_error += error_u**2 * det_J * weight
                h1_error += (error_u**2 + error_du_dx**2 + error_du_dy**2) * det_J * weight
        
        return np.sqrt(l2_error), np.sqrt(h1_error)

def analytical_verification():
    """
    Verify the exact solution satisfies the PDE
    """
    print("=== Analytical Verification ===")
    
    # Test points
    x_test = np.array([0.25, 0.5, 0.75])
    y_test = np.array([0.25, 0.5, 0.75])
    
    omega_x = omega_y = 4.0 * np.pi
    
    print("Verifying that exact solution satisfies -∇²u = f")
    print("Exact solution: u(x,y) = sin(4πx)sin(4πy)")
    print("∇²u = -2(4π)²sin(4πx)sin(4πy)")
    print("Expected RHS: f(x,y) = 2(4π)²sin(4πx)sin(4πy)")
    
    max_error = 0
    for x in x_test:
        for y in y_test:
            # Exact solution
            u_exact = exact_solution(x, y)
            
            # Second derivatives (Laplacian of sin(ωx)sin(ωy))
            d2u_dx2 = -(omega_x**2) * np.sin(omega_x * x) * np.sin(omega_y * y)
            d2u_dy2 = -(omega_y**2) * np.sin(omega_x * x) * np.sin(omega_y * y)
            laplacian = d2u_dx2 + d2u_dy2  # This is -2ω²sin(ωx)sin(ωy)
            
            # RHS function
            f_val = rhs_function(x, y)
            
            # Check if -∇²u = f
            error = abs(-laplacian - f_val)
            max_error = max(max_error, error)
            
            if x == 0.25 and y == 0.25:  # Print details for first point
                print(f"Detailed check at ({x:.2f},{y:.2f}):")
                print(f"  u = {u_exact:.6f}")
                print(f"  ∇²u = {laplacian:.6f}")
                print(f"  -∇²u = {-laplacian:.6f}")
                print(f"  f = {f_val:.6f}")
                print(f"  error = {error:.2e}")
    
    print(f"Maximum error in PDE satisfaction: {max_error:.2e}")
    return max_error < 1e-12

def test_element_integration():
    """
    Test element integration accuracy
    """
    print("\n=== Element Integration Test ===")
    
    # Create a simple 1x1 element
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    element = BilinearQuadElement()
    
    # Test integration of constant function (should equal area = 1)
    integral = 0.0
    for i, (xi, eta) in enumerate(element.gauss_points):
        weight = element.gauss_weights[i]
        J, det_J = element.jacobian(nodes, xi, eta)
        integral += 1.0 * det_J * weight
    
    print(f"Integration of constant 1 over unit square: {integral:.10f}")
    print(f"Expected: 1.0, Error: {abs(integral - 1.0):.2e}")
    
    # Test integration of linear function x
    integral_x = 0.0
    for i, (xi, eta) in enumerate(element.gauss_points):
        weight = element.gauss_weights[i]
        N = element.shape_functions(xi, eta)
        J, det_J = element.jacobian(nodes, xi, eta)
        x_gauss = np.sum(N * nodes[:, 0])
        integral_x += x_gauss * det_J * weight
    
    print(f"Integration of x over unit square: {integral_x:.10f}")
    print(f"Expected: 0.5, Error: {abs(integral_x - 0.5):.2e}")
    
    return abs(integral - 1.0) < 1e-12 and abs(integral_x - 0.5) < 1e-12

def run_verification():
    """
    Run comprehensive verification of the FEM implementation
    """
    print("=" * 60)
    print("FEM IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    # 1. Analytical verification
    analytical_ok = analytical_verification()
    
    # 2. Element integration test
    integration_ok = test_element_integration()
    
    # 3. Small mesh solution test
    print("\n=== Small Mesh Solution Test ===")
    mesh = QuadrilateralMesh(4, 4)  # 4x4 elements
    solver = PoissonFEMSolver(mesh)
    
    print(f"Mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes")
    print(f"Boundary nodes: {len(mesh.boundary_nodes)}")
    
    # Check mesh connectivity
    print("Sample element connectivity:")
    for i in range(min(3, len(mesh.elements))):
        elem = mesh.elements[i]
        coords = mesh.nodes[elem]
        print(f"  Element {i}: nodes {elem}, coords:")
        for j, coord in enumerate(coords):
            print(f"    Node {elem[j]}: ({coord[0]:.3f}, {coord[1]:.3f})")
    
    # Solve
    solution = solver.solve()
    l2_error, h1_error = solver.compute_error()
    
    print(f"\nSolution statistics:")
    print(f"  Min value: {np.min(solution):.6f}")
    print(f"  Max value: {np.max(solution):.6f}")
    print(f"  L2 error: {l2_error:.6e}")
    print(f"  H1 error: {h1_error:.6e}")
    
    # Check boundary conditions
    boundary_values = solution[mesh.boundary_nodes]
    max_boundary_error = np.max(np.abs(boundary_values))
    print(f"  Max boundary condition error: {max_boundary_error:.2e}")
    
    # 4. Convergence test
    print("\n=== Detailed Convergence Test ===")
    mesh_sizes = [4, 8, 16, 32]
    errors = []
    h_values = []
    
    for nx in mesh_sizes:
        mesh = QuadrilateralMesh(nx, nx)
        solver = PoissonFEMSolver(mesh)
        solution = solver.solve()
        l2_error, h1_error = solver.compute_error()
        errors.append(l2_error)
        h_values.append(1.0/nx)
        print(f"  {nx}x{nx} mesh (h={1.0/nx:.4f}): L2 error = {l2_error:.6e}")
    
    # Check convergence rates between consecutive meshes
    convergence_rates = []
    for i in range(1, len(errors)):
        rate = np.log(errors[i-1]/errors[i]) / np.log(h_values[i-1]/h_values[i])
        convergence_rates.append(rate)
        print(f"  Convergence rate {mesh_sizes[i-1]}→{mesh_sizes[i]}: {rate:.3f}")
    
    avg_convergence_rate = np.mean(convergence_rates)
    print(f"  Average convergence rate: {avg_convergence_rate:.3f} (expected ≈ 2.0)")
    
    # More lenient check for convergence
    convergence_ok = avg_convergence_rate > 1.7  # Allow some numerical error
    
    # Additional diagnostic: check if solution is reasonable
    print(f"\n=== Solution Diagnostics (32x32 mesh) ===")
    mesh = QuadrilateralMesh(32, 32)
    solver = PoissonFEMSolver(mesh)
    solution = solver.solve()
    
    # Check solution at center
    center_node = mesh.n_nodes // 2 + (mesh.nx + 1) // 2
    center_coords = mesh.nodes[center_node]
    center_solution = solution[center_node]
    center_exact = exact_solution(center_coords[0], center_coords[1])
    
    print(f"  Center point ({center_coords[0]:.3f}, {center_coords[1]:.3f}):")
    print(f"    FEM solution: {center_solution:.6f}")
    print(f"    Exact solution: {center_exact:.6f}")
    print(f"    Error: {abs(center_solution - center_exact):.6e}")
    
    # Check maximum values
    max_fem = np.max(np.abs(solution))
    max_exact = 1.0  # Maximum of sin(4πx)sin(4πy) is 1
    print(f"  Maximum absolute values:")
    print(f"    FEM: {max_fem:.6f}")
    print(f"    Exact: {max_exact:.6f}")
    print(f"    Ratio: {max_fem/max_exact:.6f}")
    
    # Check if boundary nodes are indeed zero
    boundary_violation = np.max(np.abs(solution[mesh.boundary_nodes]))
    print(f"  Boundary condition violation: {boundary_violation:.2e}")
    
    max_boundary_error = boundary_violation
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✓ Analytical verification: {'PASS' if analytical_ok else 'FAIL'}")
    print(f"✓ Element integration: {'PASS' if integration_ok else 'FAIL'}")
    print(f"✓ Boundary conditions: {'PASS' if max_boundary_error < 1e-12 else 'FAIL'}")
    print(f"✓ Convergence rate: {'PASS' if convergence_ok else 'FAIL'}")
    
    overall_pass = analytical_ok and integration_ok and max_boundary_error < 1e-12 and convergence_ok
    print(f"\nOVERALL: {'PASS - Implementation appears correct!' if overall_pass else 'FAIL - Issues detected'}")

    plt.figure(figsize=(10, 5))
    plt.plot(mesh_sizes, errors, marker='o', label='L2 Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mesh Size (nx)')
    plt.ylabel('L2 Error')
    plt.title('Convergence Study')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergence_study.png')

    plt.close()
    
    solution_plot = np.reshape(solution, (mesh.ny + 1, mesh.nx + 1))
    plt.figure(figsize=(8, 6))
    plt.imshow(solution_plot, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(label='FEM Solution')
    plt.title('FEM Solution Heatmap')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('fem_solution_heatmap.png')

    plt.close()
    
    return overall_pass

if __name__ == "__main__":
    run_verification()