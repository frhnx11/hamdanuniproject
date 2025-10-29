"""
Geometry Definitions for Boundary Value Problems

This module defines various geometric domains (circle, rectangle, etc.)
and provides methods for boundary checking, domain membership, and
mesh generation.
"""

import numpy as np


class Geometry:
    """Base class for geometric domains."""

    def __init__(self, name):
        self.name = name

    def is_inside(self, x, y):
        """Check if point(s) are inside the domain."""
        raise NotImplementedError

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the boundary."""
        raise NotImplementedError

    def get_bounds(self):
        """Get bounding box [xmin, xmax, ymin, ymax]."""
        raise NotImplementedError

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid for the domain."""
        raise NotImplementedError


class Circle(Geometry):
    """
    Circular domain centered at (cx, cy) with radius r.

    Parameters:
    -----------
    cx : float
        x-coordinate of center (default: 0)
    cy : float
        y-coordinate of center (default: 0)
    radius : float
        Radius of the circle (default: 1)
    """

    def __init__(self, cx=0.0, cy=0.0, radius=1.0):
        super().__init__("Circle")
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def is_inside(self, x, y):
        """Check if point(s) are inside the circle."""
        dist = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return dist < self.radius

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the circle boundary."""
        dist = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return np.abs(dist - self.radius) < tol

    def get_bounds(self):
        """Get bounding box."""
        margin = 0.1 * self.radius
        return [
            self.cx - self.radius - margin,
            self.cx + self.radius + margin,
            self.cy - self.radius - margin,
            self.cy + self.radius + margin
        ]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the circle."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_boundary_points(self, n=100):
        """Get points on the circle boundary."""
        theta = np.linspace(0, 2*np.pi, n)
        x = self.cx + self.radius * np.cos(theta)
        y = self.cy + self.radius * np.sin(theta)
        return x, y

    def __str__(self):
        return f"Circle(center=({self.cx}, {self.cy}), radius={self.radius})"


class Rectangle(Geometry):
    """
    Rectangular domain [xmin, xmax] × [ymin, ymax].

    Parameters:
    -----------
    xmin, xmax : float
        x-extent of rectangle
    ymin, ymax : float
        y-extent of rectangle
    """

    def __init__(self, xmin=-1.0, xmax=1.0, ymin=-0.5, ymax=0.5):
        super().__init__("Rectangle")
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def is_inside(self, x, y):
        """Check if point(s) are inside the rectangle."""
        return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax)

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the rectangle boundary."""
        on_left = np.abs(x - self.xmin) < tol
        on_right = np.abs(x - self.xmax) < tol
        on_bottom = np.abs(y - self.ymin) < tol
        on_top = np.abs(y - self.ymax) < tol

        in_x_range = (x >= self.xmin - tol) & (x <= self.xmax + tol)
        in_y_range = (y >= self.ymin - tol) & (y <= self.ymax + tol)

        return ((on_left | on_right) & in_y_range) | ((on_bottom | on_top) & in_x_range)

    def get_bounds(self):
        """Get bounding box."""
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        margin_x = 0.1 * width
        margin_y = 0.1 * height
        return [
            self.xmin - margin_x,
            self.xmax + margin_x,
            self.ymin - margin_y,
            self.ymax + margin_y
        ]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the rectangle."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_boundary_points(self, n=100):
        """Get points on the rectangle boundary."""
        # Distribute points along the perimeter
        perimeter = 2 * ((self.xmax - self.xmin) + (self.ymax - self.ymin))
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        n_bottom = int(n * width / perimeter)
        n_right = int(n * height / perimeter)
        n_top = int(n * width / perimeter)
        n_left = n - n_bottom - n_right - n_top

        # Bottom edge
        x_bottom = np.linspace(self.xmin, self.xmax, n_bottom, endpoint=False)
        y_bottom = np.full(n_bottom, self.ymin)

        # Right edge
        x_right = np.full(n_right, self.xmax)
        y_right = np.linspace(self.ymin, self.ymax, n_right, endpoint=False)

        # Top edge
        x_top = np.linspace(self.xmax, self.xmin, n_top, endpoint=False)
        y_top = np.full(n_top, self.ymax)

        # Left edge
        x_left = np.full(n_left, self.xmin)
        y_left = np.linspace(self.ymax, self.ymin, n_left, endpoint=False)

        x = np.concatenate([x_bottom, x_right, x_top, x_left])
        y = np.concatenate([y_bottom, y_right, y_top, y_left])

        return x, y

    def __str__(self):
        return f"Rectangle([{self.xmin}, {self.xmax}] × [{self.ymin}, {self.ymax}])"


class Square(Rectangle):
    """
    Square domain centered at (cx, cy) with side length 'side'.

    This is a special case of Rectangle.

    Parameters:
    -----------
    cx : float
        x-coordinate of center (default: 0)
    cy : float
        y-coordinate of center (default: 0)
    side : float
        Side length of the square (default: 2, giving unit square [-1,1]×[-1,1])
    """

    def __init__(self, cx=0.0, cy=0.0, side=2.0):
        half_side = side / 2.0
        super().__init__(
            xmin=cx - half_side,
            xmax=cx + half_side,
            ymin=cy - half_side,
            ymax=cy + half_side
        )
        self.name = "Square"
        self.cx = cx
        self.cy = cy
        self.side = side

    def __str__(self):
        return f"Square(center=({self.cx}, {self.cy}), side={self.side})"


class LShape(Geometry):
    """
    L-shaped domain (useful for testing corner singularities).

    The L-shape is constructed as a unit square [0,1]×[0,1] with
    the upper-right quadrant [0.5,1]×[0.5,1] removed.

    Parameters:
    -----------
    scale : float
        Scaling factor for the L-shape (default: 1.0)
    """

    def __init__(self, scale=1.0):
        super().__init__("L-Shape")
        self.scale = scale

    def is_inside(self, x, y):
        """Check if point(s) are inside the L-shape."""
        # Inside unit square [0, scale] × [0, scale]
        in_square = (x >= 0) & (x <= self.scale) & (y >= 0) & (y <= self.scale)

        # Not in upper-right quadrant [scale/2, scale] × [scale/2, scale]
        half = self.scale / 2.0
        in_removed = (x > half) & (y > half)

        return in_square & np.logical_not(in_removed)

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the L-shape boundary."""
        half = self.scale / 2.0

        # Outer boundary of square
        on_outer = ((np.abs(x) < tol) | (np.abs(x - self.scale) < tol) |
                    (np.abs(y) < tol) | (np.abs(y - self.scale) < tol))

        # Inner corner edges
        on_inner_horiz = (np.abs(y - half) < tol) & (x >= half - tol) & (x <= self.scale + tol)
        on_inner_vert = (np.abs(x - half) < tol) & (y >= half - tol) & (y <= self.scale + tol)
        on_inner = on_inner_horiz | on_inner_vert

        return (on_outer | on_inner) & self.is_inside(x, y)

    def get_bounds(self):
        """Get bounding box."""
        margin = 0.1 * self.scale
        return [-margin, self.scale + margin, -margin, self.scale + margin]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the L-shape."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def __str__(self):
        return f"L-Shape(scale={self.scale})"


class Annulus(Geometry):
    """
    Annulus (circular ring) domain with inner and outer radius.

    The annulus is the region between two concentric circles.

    Parameters:
    -----------
    cx : float
        x-coordinate of center (default: 0)
    cy : float
        y-coordinate of center (default: 0)
    r_inner : float
        Inner radius (default: 0.5)
    r_outer : float
        Outer radius (default: 1.0)
    """

    def __init__(self, cx=0.0, cy=0.0, r_inner=0.5, r_outer=1.0):
        super().__init__("Annulus")
        self.cx = cx
        self.cy = cy
        self.r_inner = r_inner
        self.r_outer = r_outer

        if r_inner >= r_outer:
            raise ValueError("Inner radius must be less than outer radius")

    def is_inside(self, x, y):
        """Check if point(s) are inside the annulus."""
        r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return (r > self.r_inner) & (r < self.r_outer)

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the annulus boundary."""
        r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        on_inner = np.abs(r - self.r_inner) < tol
        on_outer = np.abs(r - self.r_outer) < tol
        return on_inner | on_outer

    def get_bounds(self):
        """Get bounding box."""
        margin = 0.1 * self.r_outer
        return [
            self.cx - self.r_outer - margin,
            self.cx + self.r_outer + margin,
            self.cy - self.r_outer - margin,
            self.cy + self.r_outer + margin
        ]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the annulus."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_boundary_points(self, n=100):
        """Get points on the annulus boundaries."""
        # Outer circle
        theta = np.linspace(0, 2*np.pi, n)
        x_outer = self.cx + self.r_outer * np.cos(theta)
        y_outer = self.cy + self.r_outer * np.sin(theta)

        # Inner circle
        x_inner = self.cx + self.r_inner * np.cos(theta)
        y_inner = self.cy + self.r_inner * np.sin(theta)

        # Return both as separate arrays (for plotting)
        return (x_outer, y_outer), (x_inner, y_inner)

    def __str__(self):
        return f"Annulus(center=({self.cx}, {self.cy}), r_inner={self.r_inner}, r_outer={self.r_outer})"


class SemiInfinitePlane(Geometry):
    """
    Semi-infinite plane: y ≥ y0 (upper half-plane).

    Parameters:
    -----------
    y0 : float
        y-coordinate of the boundary line (default: 0)
    """

    def __init__(self, y0=0.0):
        super().__init__("Semi-Infinite Plane")
        self.y0 = y0

    def is_inside(self, x, y):
        """Check if point(s) are in the upper half-plane."""
        return y > self.y0

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the boundary line."""
        return np.abs(y - self.y0) < tol

    def get_bounds(self):
        """Get bounding box for visualization."""
        return [-2.0, 2.0, self.y0 - 0.2, self.y0 + 2.2]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the half-plane."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_boundary_points(self, n=100):
        """Get points on the boundary line."""
        xmin, xmax, _, _ = self.get_bounds()
        x = np.linspace(xmin, xmax, n)
        y = np.full(n, self.y0)
        return x, y

    def __str__(self):
        return f"Semi-Infinite Plane (y ≥ {self.y0})"


class WedgeSector(Geometry):
    """
    Wedge or angular sector with opening angle.

    The sector is defined by two rays from the origin with an angle between them.

    Parameters:
    -----------
    cx : float
        x-coordinate of apex (default: 0)
    cy : float
        y-coordinate of apex (default: 0)
    angle_start : float
        Starting angle in radians (default: 0)
    angle_end : float
        Ending angle in radians (default: π/2, giving 90° wedge)
    max_radius : float
        Maximum radius for visualization (default: 2.0)
    """

    def __init__(self, cx=0.0, cy=0.0, angle_start=0.0, angle_end=np.pi/2, max_radius=2.0):
        super().__init__("Wedge/Sector")
        self.cx = cx
        self.cy = cy
        self.angle_start = angle_start
        self.angle_end = angle_end
        self.max_radius = max_radius

        if angle_end <= angle_start:
            raise ValueError("Ending angle must be greater than starting angle")

    def is_inside(self, x, y):
        """Check if point(s) are inside the wedge."""
        # Translate to apex
        dx = x - self.cx
        dy = y - self.cy

        # Compute angle
        theta = np.arctan2(dy, dx)

        # Normalize angles to [0, 2π]
        theta = np.mod(theta, 2*np.pi)
        start = np.mod(self.angle_start, 2*np.pi)
        end = np.mod(self.angle_end, 2*np.pi)

        # Check if angle is within range
        if end > start:
            angle_ok = (theta >= start) & (theta <= end)
        else:  # Wraps around 0
            angle_ok = (theta >= start) | (theta <= end)

        # Check radius
        r = np.sqrt(dx**2 + dy**2)
        radius_ok = (r > 0) & (r < self.max_radius)

        return angle_ok & radius_ok

    def is_on_boundary(self, x, y, tol=1e-6):
        """Check if point(s) are on the wedge boundary."""
        dx = x - self.cx
        dy = y - self.cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        theta = np.mod(theta, 2*np.pi)

        # On starting ray
        on_start = np.abs(np.mod(theta - self.angle_start, 2*np.pi)) < tol
        # On ending ray
        on_end = np.abs(np.mod(theta - self.angle_end, 2*np.pi)) < tol
        # On arc
        on_arc = np.abs(r - self.max_radius) < tol

        return (on_start | on_end | on_arc) & (r <= self.max_radius)

    def get_bounds(self):
        """Get bounding box."""
        margin = 0.2 * self.max_radius
        return [
            self.cx - self.max_radius - margin,
            self.cx + self.max_radius + margin,
            self.cy - self.max_radius - margin,
            self.cy + self.max_radius + margin
        ]

    def create_mesh(self, nx=100, ny=100):
        """Create a mesh grid covering the wedge."""
        xmin, xmax, ymin, ymax = self.get_bounds()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_boundary_points(self, n=100):
        """Get points on the wedge boundary."""
        # Starting ray
        r_ray = np.linspace(0, self.max_radius, n//3)
        x_start = self.cx + r_ray * np.cos(self.angle_start)
        y_start = self.cy + r_ray * np.sin(self.angle_start)

        # Arc
        theta_arc = np.linspace(self.angle_start, self.angle_end, n//3)
        x_arc = self.cx + self.max_radius * np.cos(theta_arc)
        y_arc = self.cy + self.max_radius * np.sin(theta_arc)

        # Ending ray (reverse direction)
        x_end = self.cx + r_ray[::-1] * np.cos(self.angle_end)
        y_end = self.cy + r_ray[::-1] * np.sin(self.angle_end)

        x = np.concatenate([x_start, x_arc, x_end])
        y = np.concatenate([y_start, y_arc, y_end])

        return x, y

    def __str__(self):
        angle_deg = np.degrees(self.angle_end - self.angle_start)
        return f"Wedge(apex=({self.cx}, {self.cy}), angle={angle_deg:.1f}°)"


# Dictionary of available geometries
AVAILABLE_GEOMETRIES = {
    'circle': {
        'class': Circle,
        'name': 'Circle',
        'description': 'Circular domain (radius 1)',
        'default_params': {}
    },
    'rectangle': {
        'class': Rectangle,
        'name': 'Rectangle',
        'description': 'Rectangular domain [-1,1] × [-0.5,0.5]',
        'default_params': {}
    },
    'square': {
        'class': Square,
        'name': 'Square',
        'description': 'Unit square [-1,1] × [-1,1]',
        'default_params': {}
    },
    'lshape': {
        'class': LShape,
        'name': 'L-Shape',
        'description': 'L-shaped domain',
        'default_params': {}
    },
    'annulus': {
        'class': Annulus,
        'name': 'Annulus',
        'description': 'Circular ring (r_inner < r < r_outer)',
        'default_params': {}
    },
    'semiplane': {
        'class': SemiInfinitePlane,
        'name': 'Semi-Infinite Plane',
        'description': 'Upper half-plane (y ≥ 0)',
        'default_params': {}
    },
    'wedge': {
        'class': WedgeSector,
        'name': 'Wedge/Sector',
        'description': 'Angular sector with opening angle',
        'default_params': {}
    }
}


def create_geometry(geom_type, **params):
    """
    Factory function to create a geometry object.

    Parameters:
    -----------
    geom_type : str
        Type of geometry ('circle', 'rectangle', 'square', 'lshape')
    **params :
        Parameters specific to the geometry type

    Returns:
    --------
    Geometry object

    Example:
    --------
    >>> circle = create_geometry('circle', radius=2.0)
    >>> rect = create_geometry('rectangle', xmin=-2, xmax=2, ymin=-1, ymax=1)
    """
    if geom_type not in AVAILABLE_GEOMETRIES:
        available = ', '.join(AVAILABLE_GEOMETRIES.keys())
        raise ValueError(f"Unknown geometry '{geom_type}'. Available: {available}")

    geom_class = AVAILABLE_GEOMETRIES[geom_type]['class']
    return geom_class(**params)


