"""
Conformal Mapping Transformations

This module implements various conformal mapping functions that preserve
angles and are used to transform complex geometries into simpler domains
where analytical solutions exist.

Conformal maps are analytic functions w = f(z) where z = x + iy.
"""

import numpy as np


def circle_to_strip(z):
    """
    Map a circle to a horizontal strip using w = -i*ln(z).

    This transformation maps the unit circle |z| = 1 to a horizontal strip.
    Points inside the circle map to the interior of the strip.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain (z = x + iy)

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points (w = u + iv)

    Notes:
    ------
    - The unit circle |z| = 1 maps to a vertical line Re(w) = 0
    - Points with |z| < 1 map to Re(w) > 0
    - The transformation has a singularity at z = 0
    """
    # Avoid log(0) singularity
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)

    # w = -i * ln(z)
    w = -1j * np.log(z)

    return w


def inverse_circle_to_strip(w):
    """
    Inverse map: strip to circle using z = exp(-iw).

    Parameters:
    -----------
    w : complex or ndarray of complex
        Points in the strip domain

    Returns:
    --------
    z : complex or ndarray of complex
        Points in the circle domain
    """
    return np.exp(-1j * w)


def mobius_transform(z, a=1, b=0, c=0, d=1):
    """
    Möbius (linear fractional) transformation: w = (az + b)/(cz + d).

    This is the most general conformal map of the extended complex plane.
    Special cases include translations, rotations, scaling, and inversions.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain
    a, b, c, d : complex, optional
        Transformation parameters (must satisfy ad - bc ≠ 0)
        Default is identity transformation (a=d=1, b=c=0)

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points

    Notes:
    ------
    - Identity: a=d=1, b=c=0
    - Translation by b: a=d=1, c=0
    - Rotation by θ: a=d=exp(iθ), b=c=0
    - Inversion: a=d=0, b=c=1 gives w = 1/z
    - Constraint: ad - bc ≠ 0 (ensures invertibility)
    """
    # Check that transformation is valid
    det = a * d - b * c
    if np.abs(det) < 1e-10:
        raise ValueError("Möbius transformation is degenerate (ad - bc = 0)")

    # Avoid division by zero
    denominator = c * z + d
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)

    w = (a * z + b) / denominator

    return w


def inverse_mobius_transform(w, a=1, b=0, c=0, d=1):
    """
    Inverse Möbius transformation: z = (dw - b)/(-cw + a).

    Parameters:
    -----------
    w : complex or ndarray of complex
        Points in the transformed domain
    a, b, c, d : complex, optional
        Original transformation parameters

    Returns:
    --------
    z : complex or ndarray of complex
        Points in the original domain
    """
    # The inverse of (az+b)/(cz+d) is (dw-b)/(-cw+a)
    return mobius_transform(w, d, -b, -c, a)


def joukowski_transform(z, scale=1.0):
    """
    Joukowski transformation: w = scale * (z + 1/z).

    This transformation is famous for mapping circles to airfoil shapes.
    It's widely used in aerodynamics.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain
    scale : float, optional
        Scaling factor (default: 1.0)

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points

    Notes:
    ------
    - A circle passing through z = ±1 maps to an airfoil shape
    - The unit circle |z| = 1 maps to a line segment [-2, 2]
    - Has a singularity at z = 0
    """
    # Avoid division by zero
    z_safe = np.where(np.abs(z) < 1e-10, 1e-10, z)

    w = scale * (z_safe + 1.0 / z_safe)

    return w


def square_map(z):
    """
    Square transformation: w = z².

    Maps the upper half-plane to the right half-plane.
    Useful for solving problems in angular sectors.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points

    Notes:
    ------
    - Maps the upper half-plane Im(z) > 0 to Re(w) > 0
    - The positive real axis maps to itself
    - The imaginary axis maps to the negative real axis
    """
    return z ** 2


def sine_map(z):
    """
    Sine transformation: w = sin(z).

    Periodic mapping useful for channel and strip problems.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points

    Notes:
    ------
    - Maps vertical strips to bounded regions
    - Period is 2π
    - Maps the strip -π/2 < Re(z) < π/2 to |Re(w)| < 1
    """
    return np.sin(z)


def custom_map(z, expression="z"):
    """
    Custom conformal mapping function evaluated from a string expression.

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the original domain
    expression : str
        Mathematical expression for the mapping (e.g., "z**2 + 1", "exp(z)", "1/z")
        Available functions: sin, cos, tan, exp, log, sqrt
        Use 'z' as the variable

    Returns:
    --------
    w : complex or ndarray of complex
        Transformed points

    Examples:
    ---------
    >>> w = custom_map(z, "z**2 + 1")
    >>> w = custom_map(z, "exp(z)")
    >>> w = custom_map(z, "sin(z) + cos(z)")

    Raises:
    -------
    ValueError : if expression is invalid or unsafe
    """
    # Safety: only allow specific numpy functions and operations
    allowed_names = {
        'z': z,
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'ln': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'conj': np.conj,
        'real': np.real,
        'imag': np.imag,
        'pi': np.pi,
        'e': np.e,
    }

    try:
        # Evaluate the expression with restricted namespace
        w = eval(expression, {"__builtins__": {}}, allowed_names)
        return w
    except Exception as e:
        raise ValueError(f"Invalid custom mapping expression: {expression}\nError: {str(e)}")


def apply_map_to_grid(X, Y, map_function, *args, **kwargs):
    """
    Apply a conformal map to a 2D grid of points.

    Parameters:
    -----------
    X : ndarray
        2D array of x-coordinates (from meshgrid)
    Y : ndarray
        2D array of y-coordinates (from meshgrid)
    map_function : callable
        Conformal mapping function to apply
    *args, **kwargs :
        Additional arguments to pass to map_function

    Returns:
    --------
    U : ndarray
        2D array of transformed x-coordinates (real part of w)
    V : ndarray
        2D array of transformed y-coordinates (imaginary part of w)
    W : ndarray of complex
        Transformed complex coordinates w = u + iv

    Example:
    --------
    >>> U, V, W = apply_map_to_grid(X, Y, circle_to_strip)
    """
    # Combine x and y into complex coordinates
    Z = X + 1j * Y

    # Apply the conformal map
    W = map_function(Z, *args, **kwargs)

    # Extract real and imaginary parts
    U = np.real(W)
    V = np.imag(W)

    return U, V, W


def identity_map(z):
    """
    Identity transformation: w = z (no transformation).

    Parameters:
    -----------
    z : complex or ndarray of complex
        Points in the domain

    Returns:
    --------
    w : complex or ndarray of complex
        Same points (w = z)
    """
    return z


# Dictionary of available conformal maps
AVAILABLE_MAPS = {
    'identity': {
        'function': identity_map,
        'name': 'Identity (No transformation)',
        'description': 'w = z',
        'inverse': identity_map
    },
    'circle_to_strip': {
        'function': circle_to_strip,
        'name': 'Circle to Strip',
        'description': 'w = -i·ln(z)',
        'inverse': inverse_circle_to_strip
    },
    'mobius': {
        'function': lambda z: mobius_transform(z, a=1, b=1, c=0, d=1),
        'name': 'Möbius (Translation)',
        'description': 'w = z + 1',
        'inverse': lambda w: mobius_transform(w, a=1, b=-1, c=0, d=1)
    },
    'inversion': {
        'function': lambda z: mobius_transform(z, a=0, b=1, c=1, d=0),
        'name': 'Inversion',
        'description': 'w = 1/z',
        'inverse': lambda w: mobius_transform(w, a=0, b=1, c=1, d=0)
    },
    'joukowski': {
        'function': joukowski_transform,
        'name': 'Joukowski',
        'description': 'w = z + 1/z',
        'inverse': None  # Inverse is multivalued
    },
    'square': {
        'function': square_map,
        'name': 'Square',
        'description': 'w = z²',
        'inverse': lambda w: np.sqrt(w)  # Principal branch
    },
    'sine': {
        'function': sine_map,
        'name': 'Sine',
        'description': 'w = sin(z)',
        'inverse': lambda w: np.arcsin(w)  # Principal branch
    },
    'custom': {
        'function': lambda z: custom_map(z, "z"),  # Default identity
        'name': 'Custom Function',
        'description': 'w = f(z) (user-defined)',
        'inverse': None
    }
}


def get_map_function(map_name):
    """
    Get a conformal mapping function by name.

    Parameters:
    -----------
    map_name : str
        Name of the map (e.g., 'circle_to_strip', 'mobius', 'joukowski')

    Returns:
    --------
    function : callable
        The conformal mapping function

    Raises:
    -------
    ValueError : if map_name is not recognized
    """
    if map_name not in AVAILABLE_MAPS:
        available = ', '.join(AVAILABLE_MAPS.keys())
        raise ValueError(f"Unknown map '{map_name}'. Available maps: {available}")

    return AVAILABLE_MAPS[map_name]['function']


def get_map_info(map_name):
    """
    Get information about a conformal map.

    Parameters:
    -----------
    map_name : str
        Name of the map

    Returns:
    --------
    dict : Information about the map (name, description, etc.)
    """
    if map_name not in AVAILABLE_MAPS:
        available = ', '.join(AVAILABLE_MAPS.keys())
        raise ValueError(f"Unknown map '{map_name}'. Available maps: {available}")

    return AVAILABLE_MAPS[map_name]


