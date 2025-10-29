"""
Green's Function Solver for 2D Laplace Equation

This module implements the Green's function method for solving
boundary value problems in 2D electrostatics.
"""

import numpy as np


def greens_function_2d_laplace(x, y, x0, y0, epsilon=1e-10):
    """
    2D Laplace Green's function for a point source.

    For the 2D Laplace equation ∇²φ = -δ(r - r₀), the Green's function is:
    G(r; r₀) = -ln(|r - r₀|) / (2π)

    Parameters:
    -----------
    x : float or ndarray
        x-coordinates where to evaluate the Green's function
    y : float or ndarray
        y-coordinates where to evaluate the Green's function
    x0 : float
        x-coordinate of the point source
    y0 : float
        y-coordinate of the point source
    epsilon : float, optional
        Small value to avoid singularity at source location (default: 1e-10)

    Returns:
    --------
    G : ndarray
        Green's function values at the specified coordinates

    Notes:
    ------
    The potential due to a point charge/source is proportional to G.
    The singularity at (x0, y0) is regularized using epsilon.
    """
    # Distance from source point
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Avoid singularity at source location
    r = np.maximum(r, epsilon)

    # 2D Laplace Green's function
    G = -np.log(r) / (2 * np.pi)

    return G


def greens_function_grounded_plane(x, y, x0, y0, plane_y=0.0, epsilon=1e-10):
    """
    2D Green's function for a grounded conducting plane at y = plane_y.

    Uses the method of images: places an image source at (x0, 2*plane_y - y0)
    with opposite sign to enforce φ = 0 on the plane.

    Parameters:
    -----------
    x : float or ndarray
        x-coordinates where to evaluate
    y : float or ndarray
        y-coordinates where to evaluate
    x0 : float
        x-coordinate of the real source
    y0 : float
        y-coordinate of the real source
    plane_y : float, optional
        y-coordinate of the grounded plane (default: 0.0)
    epsilon : float, optional
        Small value to avoid singularity

    Returns:
    --------
    G : ndarray
        Green's function with boundary condition φ = 0 at y = plane_y

    Notes:
    ------
    For a grounded plane at y = 0:
    - Real source at (x0, y0)
    - Image source at (x0, -y0) with strength -1
    - G = G_real - G_image
    """
    # Real source contribution
    r_real = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_real = np.maximum(r_real, epsilon)
    G_real = -np.log(r_real) / (2 * np.pi)

    # Image source at mirror position
    y0_image = 2 * plane_y - y0
    r_image = np.sqrt((x - x0)**2 + (y - y0_image)**2)
    r_image = np.maximum(r_image, epsilon)
    G_image = -np.log(r_image) / (2 * np.pi)

    # Total Green's function (subtract image to enforce φ = 0 on plane)
    G = G_real - G_image

    return G


def greens_function_circular_region(x, y, x0, y0, radius=1.0, epsilon=1e-10):
    """
    2D Green's function for a grounded circular conducting boundary.

    Uses the method of images with inversion: places an image source at
    the inverse point with appropriate strength to enforce φ = 0 on the circle.

    Parameters:
    -----------
    x : float or ndarray
        x-coordinates where to evaluate
    y : float or ndarray
        y-coordinates where to evaluate
    x0 : float
        x-coordinate of the real source
    y0 : float
        y-coordinate of the real source
    radius : float, optional
        Radius of the circular boundary (default: 1.0)
    epsilon : float, optional
        Small value to avoid singularity

    Returns:
    --------
    G : ndarray
        Green's function with boundary condition φ = 0 at r = radius

    Notes:
    ------
    For a grounded circle of radius R centered at origin:
    - Real source at (x0, y0) with r0 = √(x0² + y0²)
    - Image source at (R²/r0² · x0, R²/r0² · y0) with strength -(R/r0)
    - This enforces φ = 0 on the circle boundary
    """
    # Distance of source from center
    r0 = np.sqrt(x0**2 + y0**2)

    if r0 < epsilon:
        # Source at center - no image needed, just regularize
        r_real = np.sqrt(x**2 + y**2)
        r_real = np.maximum(r_real, epsilon)
        return -np.log(r_real) / (2 * np.pi)

    # Real source contribution
    r_real = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_real = np.maximum(r_real, epsilon)
    G_real = -np.log(r_real) / (2 * np.pi)

    # Image source location (inversion with respect to circle)
    scale = (radius ** 2) / (r0 ** 2)
    x0_image = scale * x0
    y0_image = scale * y0

    # Image source strength
    strength_image = radius / r0

    # Image source contribution
    r_image = np.sqrt((x - x0_image)**2 + (y - y0_image)**2)
    r_image = np.maximum(r_image, epsilon)
    G_image = -np.log(r_image) / (2 * np.pi)

    # Total Green's function
    G = G_real - strength_image * G_image

    return G


def compute_potential(X, Y, sources, boundary_value=0.0, green_function=None):
    """
    Compute the electric potential due to multiple point sources.

    The total potential is the superposition of individual source contributions.

    Parameters:
    -----------
    X : ndarray
        2D array of x-coordinates (from meshgrid)
    Y : ndarray
        2D array of y-coordinates (from meshgrid)
    sources : list of tuples
        List of (x0, y0, strength) tuples for each source
        strength is the source magnitude (e.g., charge)
    boundary_value : float, optional
        Constant potential at boundary (default: 0.0)
    green_function : callable, optional
        Green's function to use. If None, uses free-space Green's function.
        Signature: green_function(X, Y, x0, y0) -> G

    Returns:
    --------
    phi : ndarray
        Total electric potential at each grid point
    """
    phi = np.zeros_like(X) + boundary_value

    # Use provided Green's function or default to free-space
    if green_function is None:
        green_function = greens_function_2d_laplace

    for x0, y0, strength in sources:
        G = green_function(X, Y, x0, y0)
        phi += strength * G

    return phi


def compute_electric_field(X, Y, sources, dx=None):
    """
    Compute the electric field E = -∇φ from the potential.

    The electric field is the negative gradient of the potential.
    Uses central difference for numerical differentiation.

    Parameters:
    -----------
    X : ndarray
        2D array of x-coordinates (from meshgrid)
    Y : ndarray
        2D array of y-coordinates (from meshgrid)
    sources : list of tuples
        List of (x0, y0, strength) tuples for each source
    dx : float, optional
        Grid spacing for numerical gradient (auto-computed if None)

    Returns:
    --------
    Ex : ndarray
        x-component of electric field
    Ey : ndarray
        y-component of electric field
    E_magnitude : ndarray
        Magnitude of electric field |E|
    """
    # Compute potential
    phi = compute_potential(X, Y, sources)

    # Compute gradient using numpy's gradient function
    # (automatically handles spacing)
    if dx is None:
        dx = X[0, 1] - X[0, 0]  # Grid spacing in x
        dy = Y[1, 0] - Y[0, 0]  # Grid spacing in y
    else:
        dy = dx

    grad_y, grad_x = np.gradient(phi, dy, dx)

    # Electric field is negative gradient of potential
    Ex = -grad_x
    Ey = -grad_y

    # Field magnitude
    E_magnitude = np.sqrt(Ex**2 + Ey**2)

    return Ex, Ey, E_magnitude


def evaluate_at_point(x, y, sources):
    """
    Evaluate potential and field at a specific point.

    Parameters:
    -----------
    x : float
        x-coordinate of evaluation point
    y : float
        y-coordinate of evaluation point
    sources : list of tuples
        List of (x0, y0, strength) tuples for each source

    Returns:
    --------
    dict with keys:
        'potential' : float
            Potential at the point
        'field_x' : float
            x-component of field (analytical)
        'field_y' : float
            y-component of field (analytical)
        'field_magnitude' : float
            Magnitude of field
    """
    potential = 0.0
    field_x = 0.0
    field_y = 0.0

    for x0, y0, strength in sources:
        # Potential
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        if r > 1e-10:  # Avoid singularity
            potential += strength * (-np.log(r) / (2 * np.pi))

            # Analytical gradient: ∂G/∂x = -(x-x0)/(2πr²)
            field_x += strength * (-(x - x0) / (2 * np.pi * r**2))
            field_y += strength * (-(y - y0) / (2 * np.pi * r**2))

    # Electric field is negative gradient
    field_x = -field_x
    field_y = -field_y
    field_magnitude = np.sqrt(field_x**2 + field_y**2)

    return {
        'potential': potential,
        'field_x': field_x,
        'field_y': field_y,
        'field_magnitude': field_magnitude
    }


def find_max_potential(phi, X, Y):
    """
    Find the maximum absolute potential value and its location.

    Parameters:
    -----------
    phi : ndarray
        2D array of potential values
    X : ndarray
        2D array of x-coordinates
    Y : ndarray
        2D array of y-coordinates

    Returns:
    --------
    dict with keys:
        'max_value' : float
            Maximum absolute potential
        'x' : float
            x-coordinate of maximum
        'y' : float
            y-coordinate of maximum
    """
    # Find location of maximum absolute value
    abs_phi = np.abs(phi)
    max_idx = np.unravel_index(np.argmax(abs_phi), phi.shape)

    return {
        'max_value': phi[max_idx],
        'x': X[max_idx],
        'y': Y[max_idx]
    }


