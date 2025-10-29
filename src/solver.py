"""
Main Solver for Electromagnetic Boundary Value Problems

This module integrates conformal mapping, Green's functions, and geometry
to solve boundary value problems in electromagnetism.
"""

import numpy as np

# Handle both relative imports (when used as module) and direct imports (when run as script)
try:
    from . import greens_solver
    from . import conformal_maps
    from . import geometry
except ImportError:
    import greens_solver
    import conformal_maps
    import geometry


class BoundaryValueSolver:
    """
    Solver for electromagnetic boundary value problems using
    conformal mapping and Green's functions.
    """

    def __init__(self, geometry_obj, conformal_map_name='identity',
                 boundary_condition='dirichlet', bc_value=0.0,
                 green_function_type='free_space'):
        """
        Initialize the solver.

        Parameters:
        -----------
        geometry_obj : Geometry
            Geometric domain object
        conformal_map_name : str
            Name of the conformal mapping to apply
        boundary_condition : str
            Type of boundary condition ('dirichlet', 'neumann', or 'mixed')
        bc_value : float
            Value of the boundary condition
        green_function_type : str
            Type of Green's function ('free_space', 'grounded_plane', 'circular_region')
        """
        self.geometry = geometry_obj
        self.conformal_map_name = conformal_map_name
        self.boundary_condition = boundary_condition
        self.bc_value = bc_value
        self.green_function_type = green_function_type

        # Get the conformal mapping function
        self.map_function = conformal_maps.get_map_function(conformal_map_name)
        self.map_info = conformal_maps.get_map_info(conformal_map_name)

        # Select Green's function based on type
        self._select_green_function()

        # Storage for results
        self.results = None

    def _select_green_function(self):
        """Select the appropriate Green's function based on type and geometry."""
        if self.green_function_type == 'free_space':
            self.green_func = greens_solver.greens_function_2d_laplace
        elif self.green_function_type == 'grounded_plane':
            # For grounded plane, get the plane location from geometry if available
            plane_y = getattr(self.geometry, 'y0', 0.0)
            self.green_func = lambda X, Y, x0, y0: greens_solver.greens_function_grounded_plane(
                X, Y, x0, y0, plane_y=plane_y
            )
        elif self.green_function_type == 'circular_region':
            # For circular region, get radius from geometry
            radius = getattr(self.geometry, 'radius', getattr(self.geometry, 'r_outer', 1.0))
            self.green_func = lambda X, Y, x0, y0: greens_solver.greens_function_circular_region(
                X, Y, x0, y0, radius=radius
            )
        else:
            raise ValueError(f"Unknown Green's function type: {self.green_function_type}")

    def solve(self, source_x, source_y, source_strength=1.0, nx=150, ny=150):
        """
        Solve the boundary value problem.

        Parameters:
        -----------
        source_x : float
            x-coordinate of the point source
        source_y : float
            y-coordinate of the point source
        source_strength : float
            Strength of the source (default: 1.0)
        nx, ny : int
            Grid resolution for visualization

        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'X', 'Y': Original domain coordinates
            - 'U', 'V': Transformed domain coordinates
            - 'phi': Electric potential in original domain
            - 'phi_transformed': Potential in transformed domain
            - 'Ex', 'Ey': Electric field components
            - 'E_magnitude': Electric field magnitude
            - 'source': Source location and properties
            - 'max_potential': Maximum potential information
            - 'field_at_source': Field at a nearby point
        """
        # Create mesh for the original domain
        X, Y = self.geometry.create_mesh(nx, ny)

        # Check if source is inside the domain
        source_inside = self.geometry.is_inside(source_x, source_y)
        if not source_inside:
            print(f"Warning: Source at ({source_x}, {source_y}) is outside the domain!")

        # Define source(s)
        sources = [(source_x, source_y, source_strength)]

        # Compute potential in the original domain using selected Green's function
        phi_original = greens_solver.compute_potential(X, Y, sources, self.bc_value,
                                                        green_function=self.green_func)

        # Apply conformal mapping
        U, V, W = conformal_maps.apply_map_to_grid(X, Y, self.map_function)

        # Transform the source location
        z_source = source_x + 1j * source_y
        w_source = self.map_function(z_source)
        source_x_transformed = np.real(w_source)
        source_y_transformed = np.imag(w_source)

        # Compute potential in the transformed domain
        sources_transformed = [(source_x_transformed, source_y_transformed, source_strength)]
        phi_transformed = greens_solver.compute_potential(U, V, sources_transformed, self.bc_value)

        # Compute electric field in original domain
        Ex, Ey, E_magnitude = greens_solver.compute_electric_field(X, Y, sources)

        # Find maximum potential
        max_pot_info = greens_solver.find_max_potential(phi_original, X, Y)

        # Evaluate field at a point near the source (not exactly at it to avoid singularity)
        eval_x = source_x + 0.1
        eval_y = source_y + 0.1
        field_info = greens_solver.evaluate_at_point(eval_x, eval_y, sources)

        # Create a mask for points outside the domain (for visualization)
        inside_mask = self.geometry.is_inside(X, Y)

        # Mask potentials outside the domain
        phi_original_masked = np.where(inside_mask, phi_original, np.nan)
        E_magnitude_masked = np.where(inside_mask, E_magnitude, np.nan)

        # Store results
        self.results = {
            # Original domain
            'X': X,
            'Y': Y,
            'phi': phi_original_masked,
            'phi_unmasked': phi_original,
            'Ex': Ex,
            'Ey': Ey,
            'E_magnitude': E_magnitude_masked,
            'inside_mask': inside_mask,

            # Transformed domain
            'U': U,
            'V': V,
            'W': W,
            'phi_transformed': phi_transformed,
            'source_transformed': (source_x_transformed, source_y_transformed),

            # Source information
            'source': {
                'x': source_x,
                'y': source_y,
                'strength': source_strength,
                'inside_domain': source_inside
            },

            # Numerical results
            'max_potential': max_pot_info,
            'field_at_point': {
                'location': (eval_x, eval_y),
                'values': field_info
            },

            # Metadata
            'geometry': str(self.geometry),
            'conformal_map': self.map_info['description'],
            'boundary_condition': f"{self.boundary_condition} = {self.bc_value}"
        }

        return self.results

    def get_summary(self):
        """
        Get a text summary of the solution.

        Returns:
        --------
        summary : str
            Formatted text summary of results
        """
        if self.results is None:
            return "No solution computed yet. Call solve() first."

        r = self.results
        summary = []
        summary.append("=" * 60)
        summary.append("ELECTROMAGNETIC BOUNDARY VALUE PROBLEM SOLUTION")
        summary.append("=" * 60)
        summary.append(f"Geometry: {r['geometry']}")
        summary.append(f"Conformal Map: {r['conformal_map']}")
        summary.append(f"Green's Function: {self.green_function_type}")
        summary.append(f"Boundary Condition: {r['boundary_condition']}")
        summary.append("")
        summary.append("SOURCE INFORMATION:")
        summary.append(f"  Location: ({r['source']['x']:.3f}, {r['source']['y']:.3f})")
        summary.append(f"  Strength: {r['source']['strength']:.3f}")
        summary.append(f"  Inside domain: {r['source']['inside_domain']}")
        summary.append("")
        summary.append("SOLUTION RESULTS:")
        summary.append(f"  Maximum potential: {r['max_potential']['max_value']:.6f}")
        summary.append(f"  Location of max: ({r['max_potential']['x']:.3f}, {r['max_potential']['y']:.3f})")
        summary.append("")
        eval_loc = r['field_at_point']['location']
        eval_vals = r['field_at_point']['values']
        summary.append(f"FIELD AT POINT ({eval_loc[0]:.3f}, {eval_loc[1]:.3f}):")
        summary.append(f"  Potential: {eval_vals['potential']:.6f}")
        summary.append(f"  Field magnitude: {eval_vals['field_magnitude']:.6f}")
        summary.append(f"  Field components: Ex={eval_vals['field_x']:.6f}, Ey={eval_vals['field_y']:.6f}")
        summary.append("")
        summary.append("TRANSFORMED DOMAIN:")
        summary.append(f"  Source in transformed coords: ({r['source_transformed'][0]:.3f}, {r['source_transformed'][1]:.3f})")
        summary.append("=" * 60)

        return "\n".join(summary)


def quick_solve(geometry_type='circle', conformal_map='identity',
                source_x=0.2, source_y=-0.3, source_strength=1.0,
                bc_type='dirichlet', bc_value=0.0,
                green_function_type='free_space', nx=150, ny=150):
    """
    Quick solve function for convenience.

    Parameters:
    -----------
    geometry_type : str
        Type of geometry ('circle', 'rectangle', 'square', 'lshape', 'annulus', 'semiplane', 'wedge')
    conformal_map : str
        Conformal mapping to apply
    source_x, source_y : float
        Source location
    source_strength : float
        Source strength
    bc_type : str
        Boundary condition type
    bc_value : float
        Boundary condition value
    green_function_type : str
        Green's function type ('free_space', 'grounded_plane', 'circular_region')
    nx, ny : int
        Grid resolution

    Returns:
    --------
    solver : BoundaryValueSolver
        Solver object with computed results
    """
    # Create geometry
    geom = geometry.create_geometry(geometry_type)

    # Create solver
    solver = BoundaryValueSolver(geom, conformal_map, bc_type, bc_value, green_function_type)

    # Solve
    solver.solve(source_x, source_y, source_strength, nx, ny)

    return solver


