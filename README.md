# ⚡ Conformal Field Solver

A web application for solving boundary value problems in electromagnetism using conformal mapping and Green's functions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

## 📋 Project Description

This interactive web application solves 2D electromagnetic boundary value problems by combining:
- **Green's Function Method**: Analytical solution for point sources
- **Conformal Mapping**: Transforms complex geometries into simpler domains
- **Interactive Visualization**: Real-time plotting of potential and field distributions

### Key Features

✅ Multiple predefined geometries (Circle, Rectangle, Square, L-Shape, Annulus, Semi-Infinite Plane, Wedge/Sector)
✅ 8 conformal mapping transformations (Identity, Circle→Strip, Möbius, Inversion, Joukowski, Square, Sine, Custom)
✅ 3 Green's function types (Free Space, Grounded Plane, Circular Region)
✅ Interactive source placement with real-time sliders
✅ Boundary condition selection (Dirichlet, Neumann)
✅ Beautiful field visualizations with customizable colormaps
✅ Export to MATLAB scripts (.m files)
✅ High-resolution plot downloads (PNG 300 DPI, SVG vector)
✅ Numerical results with max potential and field values

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `streamlit` - Web interface

### Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The app will automatically open in your web browser at `http://localhost:8501`

## 📖 How to Use

1. **Select Geometry**: Choose from Circle, Rectangle, Square, L-Shape, Annulus, Semi-Infinite Plane, or Wedge/Sector
2. **Choose Conformal Map**: Select a transformation (try Identity first to understand the basics)
3. **Select Green's Function**: Choose Free Space, Grounded Plane, or Circular Region
4. **Set Source Location**: Use sliders to place the electromagnetic point source (x₀, y₀)
5. **Configure Parameters**: Adjust source strength and boundary conditions
6. **Compute Solution**: Click the "🔍 Compute Solution" button
7. **View Results**: See potential distribution, electric field, and numerical values
8. **Download**: Export MATLAB scripts and high-resolution plots

### Example Workflow

**Scenario**: Point charge in a circular conductor
1. Geometry: Circle
2. Conformal Map: Identity (no transformation)
3. Source: x₀ = 0.2, y₀ = -0.3
4. Boundary Condition: Dirichlet (V = 0)
5. Click "Compute Solution"
6. Download MATLAB script and PNG plots

## 📁 Project Structure

```
conformal-field-solver/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── src/                        # Core computational modules
│   ├── __init__.py
│   ├── greens_solver.py        # Green's function implementation
│   ├── conformal_maps.py       # Conformal transformation functions
│   ├── geometry.py             # Geometry definitions (Circle, Rectangle, etc.)
│   ├── solver.py               # Main solver integrating all components
│   └── matlab_generator.py     # MATLAB script generation
│
└── tests/                      # Unit tests
    └── __init__.py
```

### Module Descriptions

**`greens_solver.py`**: 2D Laplace Green's function G(x,y; x₀,y₀) = -ln(r)/(2π) and electric field computation

**`conformal_maps.py`**: Conformal transformations (Circle→Strip, Möbius, Joukowski, etc.)

**`geometry.py`**: Domain definitions with boundary checking and mesh generation

**`solver.py`**: Combines geometry + conformal mapping + Green's functions to solve BVPs

**`matlab_generator.py`**: Generates standalone MATLAB .m scripts from solutions

**`app.py`**: Streamlit web interface with interactive controls and visualization

## 📐 Mathematical Background

### Green's Function Method

The 2D Laplace equation with a point source:
```
∇²φ = -δ(r - r₀)
```

Has the fundamental solution (Green's function):
```
G(x, y; x₀, y₀) = -ln(r) / (2π)

where r = √[(x - x₀)² + (y - y₀)²]
```

The electric potential φ due to a source of strength q at (x₀, y₀) is:
```
φ(x, y) = q · G(x, y; x₀, y₀)
```

The electric field is the negative gradient:
```
E = -∇φ
```

### Conformal Mapping

Conformal maps w = f(z) preserve angles and transform complex geometries into simpler domains where analytical solutions are easier to obtain.

**Available Transformations:**
- **Identity**: w = z (no transformation)
- **Circle → Strip**: w = -i·ln(z)
- **Möbius**: w = (az + b)/(cz + d) (includes translations, rotations, inversions)
- **Inversion**: w = 1/z
- **Joukowski**: w = z + 1/z (used in airfoil theory)
- **Square**: w = z²
- **Sine**: w = sin(z)
- **Custom**: User-defined function expression

### Boundary Conditions

- **Dirichlet**: φ = constant on boundary (conducting surface at fixed potential)
- **Neumann**: ∂φ/∂n = constant on boundary (fixed normal electric field)

## 🔧 Troubleshooting

**Q: Source is outside the domain**
- A: The app will warn you. Move the source inside using the sliders or the solution may not be physical.

**Q: Plots look strange or have singularities**
- A: This is normal near the source location. The Green's function has a logarithmic singularity at r = 0.

**Q: MATLAB script doesn't run**
- A: Ensure you have MATLAB R2016b or later. Check that all functions are defined (they're included in the script).

**Q: Low resolution / slow performance**
- A: Reduce grid resolution in sidebar (try "Medium (100)" instead of "Very High (200)")

## 🎓 University Project

**Course**: Electromagnetism / Mathematical Physics
**Topic**: Solving Boundary Value Problems in Electromagnetism using Conformal Mapping and Green's Function

### Academic References

1. Jackson, J.D. - *Classical Electrodynamics* (Chapter 2: Boundary Value Problems)
2. Churchill, R.V. - *Complex Variables and Applications* (Conformal Mapping)
3. Morse, P.M. & Feshbach, H. - *Methods of Theoretical Physics* (Green's Functions)

## 📝 License

Educational use only - University Project
Not intended for commercial use

---

**Developed as part of a university electromagnetics course project**
