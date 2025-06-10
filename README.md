# 2D Poisson Equation Solver: Classical vs Neural Methods

A comprehensive comparison of traditional finite difference methods and modern neural operator approaches for solving the 2D Poisson equation with Dirichlet boundary conditions.

## 🔬 Problem Description

This repository implements solvers for the 2D Poisson equation:
```
-∇²u = f(x,y)  in Ω = [0,1]²
u = 0           on ∂Ω
```

Where:
- `f(x,y) = -2(4π)² sin(4πx) sin(4πy)` (right-hand side)
- `u(x,y) = sin(4πx) sin(4πy)` (exact solution for validation)

## 🚀 Features

### Classical Finite Difference Method
- **5-point stencil** implementation for the 2D Laplacian
- **Sparse matrix** formulation for efficient linear system solving
- **Convergence analysis** with L2 and H1 error norms
- **Multi-grid resolution** testing (17×17 to 129×129)
- **Theoretical convergence rates**: O(h²) for L2, O(h) for H1

### Fourier Neural Operator (FNO)
- **Spectral convolution layers** operating in Fourier domain
- **Parameter-efficient** architecture with ~32k parameters
- **Resolution-invariant** predictions
- **End-to-end learning** from PDE data

### Visualization & Analysis
- **Solution comparison plots** (exact vs numerical vs error)
- **Convergence study visualizations**
- **Training history tracking** for neural methods
- **Error metrics** and performance benchmarking

## 📊 Key Results

| Method | Grid Size | L2 Error | H1 Error | Solve Time |
|--------|-----------|----------|----------|------------|
| FD | 65×65 | ~1e-4 | ~1e-2 | ~0.1s |
| FD | 129×129 | ~2e-5 | ~5e-3 | ~0.8s |
| FNO | 64×64 | ~1e-3* | ~1e-2* | ~0.01s |

*Results depend on training quality and data

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/2d-poisson-solver.git
cd 2d-poisson-solver

# Install dependencies
pip install torch numpy scipy matplotlib
```

## 📁 Repository Structure

```
2d-poisson-solver/
├── src 
    ├── fno_model.py           # Fourier Neural Operator implementation
    ├── finite_difference.py   # Classical FD solver with convergence study
    ├── plotting_utils.py      # Visualization utilities
    ├── common_functions.py    # Shared mathematical functions
    ├── train_fno.py          # FNO training script
    ├── examples/             # Example notebooks and scripts
    ├── results/              # Generated plots and data
└── README.md
```

## 🏃‍♂️ Quick Start

### Run Finite Difference Solver
```python
python finite_difference.py
```

### Train and Test FNO
```python
python train_fno.py
```

### Compare Both Methods
```python
from finite_difference import PoissonFDSolver
from fno_model import FNO2d, get_default_fno_config

# Classical method
fd_solver = PoissonFDSolver(nx=65, ny=65)
u_fd, u_exact, X, Y, l2_err, h1_err = fd_solver.solve()

# Neural method (after training)
fno_model = FNO2d(modes1=12, modes2=12, width=32)
# ... load trained weights and predict
```

## 📈 Performance Comparison

### Computational Complexity
- **FD Method**: O(N²) memory, O(N²) solve time for N×N grid
- **FNO Method**: O(1) parameters, O(N² log N) inference time

### Accuracy vs Speed Trade-offs
- **FD**: High accuracy, moderate speed, scales with grid resolution
- **FNO**: Moderate accuracy, very fast inference, resolution-invariant

### Use Cases
- **FD**: High-precision single solutions, academic research
- **FNO**: Real-time applications, parametric studies, uncertainty quantification

## 🧮 Mathematical Background

### Finite Difference Discretization
The 2D Laplacian is discretized using the 5-point stencil:
```
    u[i,j-1]
u[i-1,j] -4u[i,j] u[i+1,j]  = f[i,j]
    u[i,j+1]
```

### Fourier Neural Operator
FNO learns the mapping between function spaces:
- **Input**: Right-hand side f(x,y) + coordinates
- **Output**: Solution u(x,y)
- **Key insight**: Convolution in Fourier domain = multiplication

## 📚 References

1. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*
2. LeVeque, R. J. "Finite Difference Methods for Ordinary and Partial Differential Equations." *SIAM 2007*
3. Trefethen, L. N. "Finite Difference and Spectral Methods for Ordinary and Partial Differential Equations." *1996*

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Multi-GPU training for FNO
- [ ] Adaptive mesh refinement for FD
- [ ] More complex boundary conditions
- [ ] 3D extension
- [ ] Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Diya Nag Chaudhury
- **Email**: diyanag@iisc.ac.in
- **Research Group**: AiREX Lab

## 🏆 Acknowledgments

- PyTorch team for the deep learning framework
- SciPy community for numerical computing tools
- FNO authors for the innovative neural operator approach

---
