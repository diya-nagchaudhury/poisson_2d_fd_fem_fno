from finite_difference_solver import main as fd_main
from fno_model import FNO2d, get_default_fno_config
from plotting_utils import plot_solution_comparison
import torch

def main():
    print("Running 2D Poisson Equation Solvers")
    print("=" * 50)
    
    # Run finite difference solver
    print("\n1. Running Finite Difference Solver...")
    u_fd, X, Y, u_exact = fd_main()
    
    # Run FNO (if you have a trained model)
    print("\n2. FNO Model Setup...")
    config = get_default_fno_config()
    fno_model = FNO2d(
        modes1=config['modes1'],
        modes2=config['modes2'],
        width=config['width'],
        layers=config['layers']
    )
    
    print("FNO model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in fno_model.parameters())}")
    
    # Compare solutions
    print("\n3. Plotting comparison...")
    plot_solution_comparison(u_fd, X, Y, u_exact, "Finite Difference")

if __name__ == "__main__":
    main()