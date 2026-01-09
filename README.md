# Monte Carlo Geometry Processing (Walk on Spheres)

A **mesh free** Python implementation of the **Walk on Spheres (WoS)** algorithm to solve linear **elliptic PDEs** (Laplace, Poisson, and extensions) directly on complex geometry.  
Unlike FEM, WoS evaluates the solution **locally at arbitrary query points** without building a volumetric mesh, which makes it especially convenient for **dirty geometry** (for example polygon soups) and smooth parametric boundaries (for example Bézier curves).

This repository contains:
* a Python WoS solver (core utilities and geometry helpers)
* example scripts
* a notebook reproducing experiments
* the project report and slides

## Key ideas

WoS replaces small Brownian motion steps by **adaptive jumps**:
at position `x`, compute the distance `R` to the boundary, then jump uniformly to the surface of the largest inscribed sphere of radius `R`. Repeat until you reach an `ε` shell near the boundary.

## Features

* Laplace solver with Dirichlet boundary conditions
* Poisson solver in 2D with a source term using a local Green function accumulation
* Variance reduction
  * control variates (pilot estimate of the gradient and first step correction)
  * (importance sampling is discussed in the report)
* Geometry processing oriented experiments
  * Polygon soup benchmark with RGB boundary values
  * Bézier curve boundaries (diffusion curves like setups)
  * Additional toy scenes (2D shapes and a 3D “reactor” SDF)

## Repository structure

* `utils.py`
  * WoS solvers (Laplace, Poisson 2D)
  * generic rendering helpers (batch sampling, control variate pipeline)
  * optional FEM baseline utilities (Delaunay + sparse solve) for comparison
* `geometry.py`
  * SDF primitives and scenes
  * polygon soup distance helpers
  * Bézier curve sampling and fast closest point queries (KD tree)
* `Random_walk.py`
  * baseline Brownian random walk (absorbing boundary)
  * WoS path generator
* `exemple.py`
  * small experiments (capacitor style boundary conditions, polygon soup color hit, 3D reactor BC)
* `Project_notebook.ipynb`
  * end to end experiments and visualizations
* `Project Report.pdf`, `Presentation.pdf`
  * theory, experiments, discussion, limitations

## Installation

Python 3.9+ recommended.

Minimal dependencies:
* numpy
* scipy
* matplotlib

Install with:
```bash
pip install numpy scipy matplotlib
```

## Quick start

### 1) Solve Laplace or Poisson at a single point (2D)

The solvers follow a simple interface:

* `sdf(x)` returns the signed distance to the boundary (negative inside, near zero on the boundary)
* `bc(x)` returns the Dirichlet boundary value at `x`
* `src(x)` (optional) returns the source term for Poisson

Example pattern:

```python
import numpy as np
from utils import wos_poisson_2d
from geometry import get_distance

def sdf_unit_square(p):
    # expects p as a 2D point, returns negative inside
    x, y = p
    return -get_distance((x, y))

def bc_zero(p):
    return 0.0

u_hat = wos_poisson_2d(
    point=np.array([0.2, 0.3]),
    sdf=sdf_unit_square,
    bc=bc_zero,
    src=None,
    n_walks=512,
    eps=1e-4,
)
print(u_hat)
```



### 2) Render an image (grid of query points)

Use the batch renderer in `utils.py` to compute a 2D field on a grid.  
The code supports scalar or vector valued boundary conditions (for example RGB).

A typical workflow is:

* define `sdf_func(points)` vectorized on an array of points shaped `(N, dim)`
* define `bc_func(points)` vectorized on `(N, dim)`
* call the rendering helper (with or without control variates)

### 3) Control variates

For faster convergence at equal sample budget, use the control variate renderer (pilot pass plus corrected pass). This often reduces Monte Carlo noise for the same compute.

## Notes on parameters

* `eps` (epsilon): termination shell tolerance near the boundary  
  Smaller `eps` reduces bias but increases the number of jumps only logarithmically
* `n_walks`: controls variance  
  Noise decreases like `1 / sqrt(n_walks)`

## Known limitations

WoS is powerful but not universal:

* Blindness in narrow tunnels (walkers may bounce for a long time, convergence can become slow)
* Gradient estimation can become unstable near the boundary (variance explosion due to `1 / R` factors)

These issues are discussed and illustrated in the report and slides.

## References

This project is based on classical links between Brownian motion and harmonic functions (Kakutani) and the Walk on Spheres method (Muller), with a modern geometry processing context inspired by:

* Sawhney and Crane, Monte Carlo Geometry Processing: A Grid Free Approach to PDE Based Methods on Volumetric Domains, ACM TOG 2020
* Sawhney et al., Grid Free Monte Carlo for PDEs with Spatially Varying Coefficients, ACM TOG 2022

## Authors

Baran Celik  
Swann Cordier  
Antoine Le Maguet


