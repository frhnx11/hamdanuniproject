"""
Conformal Field Solver - Streamlit Web Application

A web interface for solving electromagnetic boundary value problems
using conformal mapping and Green's functions.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

from src import solver, geometry, conformal_maps, matlab_generator


# Page configuration
st.set_page_config(
    page_title="Conformal Field Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's menu and make sidebar permanently fixed
hide_streamlit_style = """
<style>
/* Hide menu and deploy button */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}
div[data-testid="stToolbar"] {display: none;}

/* Hide ONLY the sidebar collapse button (not other buttons like Compute) */
button[kind="header"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}
[data-testid="baseButton-header"] {display: none !important;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Title and description
st.title("⚡ Conformal Field Solver")
st.markdown("""
Solve electromagnetic boundary value problems using **conformal mapping** and **Green's functions**.
""")

# Sidebar inputs
st.sidebar.header("Problem Setup")

# Geometry selection
st.sidebar.subheader("1. Select Geometry")
geometry_options = {
    'Circle': 'circle',
    'Rectangle': 'rectangle',
    'Square': 'square',
    'L-Shape': 'lshape',
    'Annulus (Ring)': 'annulus',
    'Semi-Infinite Plane': 'semiplane',
    'Wedge/Sector': 'wedge'
}
selected_geometry_display = st.sidebar.selectbox(
    "Domain geometry:",
    options=list(geometry_options.keys()),
    index=0
)
selected_geometry = geometry_options[selected_geometry_display]

# Conformal map selection
st.sidebar.subheader("2. Choose Conformal Map")
map_options = {}
for key, info in conformal_maps.AVAILABLE_MAPS.items():
    map_options[f"{info['name']} ({info['description']})"] = key

selected_map_display = st.sidebar.selectbox(
    "Conformal transformation:",
    options=list(map_options.keys()),
    index=0
)
selected_map = map_options[selected_map_display]

# Custom function input (if custom map selected)
custom_expression = "z"
if selected_map == 'custom':
    custom_expression = st.sidebar.text_input(
        "Custom mapping expression:",
        value="z**2",
        help="Enter expression using 'z'. Available: sin, cos, exp, log, sqrt, etc."
    )

# Green's function selection
st.sidebar.subheader("2b. Green's Function Type")
green_func_options = {
    'Free Space': 'free_space',
    'Grounded Plane (Method of Images)': 'grounded_plane',
    'Circular Region (Method of Images)': 'circular_region'
}
selected_green_display = st.sidebar.selectbox(
    "Green's function:",
    options=list(green_func_options.keys()),
    index=0,
    help="Free space: standard Green's function. Grounded plane: enforces V=0 at y=0. Circular: enforces V=0 on circle boundary."
)
selected_green = green_func_options[selected_green_display]

# Source location
st.sidebar.subheader("3. Set Source Location")
source_x = st.sidebar.slider(
    "Source x-coordinate (x₀):",
    min_value=-1.5,
    max_value=1.5,
    value=0.2,
    step=0.05
)
source_y = st.sidebar.slider(
    "Source y-coordinate (y₀):",
    min_value=-1.5,
    max_value=1.5,
    value=-0.3,
    step=0.05
)

# Source strength
source_strength = st.sidebar.slider(
    "Source strength:",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

# Boundary condition
st.sidebar.subheader("4. Boundary Condition")
bc_options = {
    'Dirichlet (V = constant on boundary)': 'dirichlet',
    'Neumann (∂V/∂n = constant on boundary)': 'neumann'
}
selected_bc_display = st.sidebar.selectbox(
    "Boundary condition type:",
    options=list(bc_options.keys()),
    index=0
)
selected_bc = bc_options[selected_bc_display]

bc_value = st.sidebar.number_input(
    "Boundary value:",
    value=0.0,
    step=0.1
)

# Grid resolution
st.sidebar.subheader("5. Visualization Settings")
resolution = st.sidebar.select_slider(
    "Grid resolution:",
    options=['Low (50)', 'Medium (100)', 'High (150)', 'Very High (200)'],
    value='High (150)'
)
resolution_map = {
    'Low (50)': 50,
    'Medium (100)': 100,
    'High (150)': 150,
    'Very High (200)': 200
}
nx = ny = resolution_map[resolution]

# Colormap selection
colormap = st.sidebar.selectbox(
    "Color scheme:",
    options=['RdBu_r', 'viridis', 'plasma', 'coolwarm', 'seismic', 'jet'],
    index=0
)

# Compute button
st.sidebar.markdown("---")
compute_button = st.sidebar.button("🔍 Compute Solution", type="primary", use_container_width=True)

# Main content area
# Initialize session state for results persistence
if 'results' not in st.session_state:
    st.session_state.results = None
if 'solver' not in st.session_state:
    st.session_state.solver = None
if 'geom' not in st.session_state:
    st.session_state.geom = None
if 'fig1' not in st.session_state:
    st.session_state.fig1 = None
if 'fig2' not in st.session_state:
    st.session_state.fig2 = None
if 'fig3' not in st.session_state:
    st.session_state.fig3 = None
if 'matlab_script' not in st.session_state:
    st.session_state.matlab_script = None

# Compute solution if button clicked
if compute_button:
    with st.spinner('Computing solution...'):
        try:
            # Create geometry object
            geom = geometry.create_geometry(selected_geometry)

            # Handle custom conformal mapping
            if selected_map == 'custom':
                # Update the custom map function with the user's expression
                conformal_maps.AVAILABLE_MAPS['custom']['function'] = lambda z: conformal_maps.custom_map(z, custom_expression)

            # Validate source location
            source_inside = geom.is_inside(source_x, source_y)
            source_on_boundary = geom.is_on_boundary(source_x, source_y)

            if source_on_boundary:
                st.warning(f"⚠️ Warning: Source is on the boundary at ({source_x:.2f}, {source_y:.2f}). "
                          "This may cause numerical instabilities.")
            elif not source_inside:
                st.warning(f"⚠️ Warning: Source is outside the domain at ({source_x:.2f}, {source_y:.2f}). "
                          "Solution will still be computed but may not be physically meaningful.")

            # Create solver with selected Green's function type
            bv_solver = solver.BoundaryValueSolver(
                geom,
                selected_map,
                selected_bc,
                bc_value,
                selected_green
            )

            # Solve the problem
            results = bv_solver.solve(source_x, source_y, source_strength, nx, ny)

            # Store ALL results in session state immediately
            st.session_state.results = results
            st.session_state.solver = bv_solver
            st.session_state.geom = geom
            st.session_state.source_inside = source_inside
            st.session_state.source_on_boundary = source_on_boundary

            # === CREATE ALL FIGURES HERE (ONCE) ===
            X, Y = results['X'], results['Y']
            inside_mask = results['inside_mask']

            # Figure 1: Domain Comparison
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Left plot: Original domain
            if hasattr(geom, 'get_boundary_points'):
                boundary_pts = geom.get_boundary_points(200)
                if isinstance(boundary_pts[0], tuple):
                    for i, (bx, by) in enumerate(boundary_pts):
                        label = 'Boundary' if i == 0 else None
                        ax1.plot(bx, by, 'k-', linewidth=2, label=label)
                else:
                    bx, by = boundary_pts
                    ax1.plot(bx, by, 'k-', linewidth=2, label='Boundary')

            ax1.plot(source_x, source_y, 'r*', markersize=20, label=f'Source ({source_x:.2f}, {source_y:.2f})')
            ax1.contourf(X, Y, inside_mask.astype(float), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.3)
            ax1.set_xlabel('x', fontsize=12)
            ax1.set_ylabel('y', fontsize=12)
            ax1.set_title('Original Domain', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_aspect('equal')

            # Right plot: Transformed domain
            U, V = results['U'], results['V']
            source_x_t, source_y_t = results['source_transformed']
            ax2.plot(source_x_t, source_y_t, 'r*', markersize=20, label=f'Source ({source_x_t:.2f}, {source_y_t:.2f})')
            ax2.scatter(U[inside_mask], V[inside_mask], c='lightblue', s=1, alpha=0.5)
            ax2.set_xlabel('u (Re w)', fontsize=12)
            ax2.set_ylabel('v (Im w)', fontsize=12)
            ax2.set_title(f'Transformed Domain: {selected_map_display.split("(")[0].strip()}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal')
            plt.tight_layout()

            # Figure 2: Potential Distribution
            fig2, ax = plt.subplots(figsize=(12, 10))
            phi = results['phi']
            levels = 20
            contour = ax.contourf(X, Y, phi, levels=levels, cmap=colormap)
            contour_lines = ax.contour(X, Y, phi, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
            cbar = plt.colorbar(contour, ax=ax, label='Potential φ')

            if hasattr(geom, 'get_boundary_points'):
                boundary_pts = geom.get_boundary_points(200)
                if isinstance(boundary_pts[0], tuple):
                    for i, (bx, by) in enumerate(boundary_pts):
                        label = 'Boundary' if i == 0 else None
                        ax.plot(bx, by, 'k-', linewidth=2, label=label)
                else:
                    bx, by = boundary_pts
                    ax.plot(bx, by, 'k-', linewidth=2, label='Boundary')

            ax.plot(source_x, source_y, 'r*', markersize=20, label='Source', zorder=10)
            max_x = results['max_potential']['x']
            max_y = results['max_potential']['y']
            ax.plot(max_x, max_y, 'wo', markersize=10, markeredgecolor='black',
                   markeredgewidth=2, label='Max Potential', zorder=10)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Electric Potential φ(x,y)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Figure 3: Electric Field
            fig3, ax = plt.subplots(figsize=(12, 10))
            Ex = results['Ex']
            Ey = results['Ey']
            E_mag = results['E_magnitude']
            contour = ax.contourf(X, Y, E_mag, levels=20, cmap='YlOrRd', alpha=0.8)
            cbar = plt.colorbar(contour, ax=ax, label='Field Magnitude |E|')

            step = max(1, nx // 20)
            ax.quiver(X[::step, ::step], Y[::step, ::step],
                     Ex[::step, ::step], Ey[::step, ::step],
                     color='blue', alpha=0.6, scale=20, width=0.003)

            if hasattr(geom, 'get_boundary_points'):
                boundary_pts = geom.get_boundary_points(200)
                if isinstance(boundary_pts[0], tuple):
                    for i, (bx, by) in enumerate(boundary_pts):
                        label = 'Boundary' if i == 0 else None
                        ax.plot(bx, by, 'k-', linewidth=2, label=label)
                else:
                    bx, by = boundary_pts
                    ax.plot(bx, by, 'k-', linewidth=2, label='Boundary')

            ax.plot(source_x, source_y, 'r*', markersize=20, label='Source', zorder=10)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Electric Field E = -∇φ', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Store figures in session state
            st.session_state.fig1 = fig1
            st.session_state.fig2 = fig2
            st.session_state.fig3 = fig3

            # Generate and cache MATLAB script
            matlab_script = matlab_generator.generate_matlab_script(
                results,
                geometry_name=selected_geometry,
                conformal_map=selected_map,
                source_x=source_x,
                source_y=source_y,
                source_strength=source_strength,
                bc_type=selected_bc,
                bc_value=bc_value,
                nx=nx,
                ny=ny
            )
            st.session_state.matlab_script = matlab_script

        except Exception as e:
            st.error(f"Error computing solution: {str(e)}")
            st.exception(e)

# Display results section - ONLY display cached results, never recompute
if st.session_state.results is not None:
    # Extract from session state
    results = st.session_state.results
    bv_solver = st.session_state.solver
    geom = st.session_state.geom
    source_inside = st.session_state.get('source_inside', True)
    source_on_boundary = st.session_state.get('source_on_boundary', False)

    # Get cached figures and MATLAB script
    fig1 = st.session_state.fig1
    fig2 = st.session_state.fig2
    fig3 = st.session_state.fig3
    matlab_script = st.session_state.matlab_script

    # Display success message
    if source_inside and not source_on_boundary:
        st.success("✓ Solution computed successfully!")
    else:
        st.info("✓ Solution computed (with warnings - see above)")

    # Display summary in an expander
    with st.expander("📊 Numerical Results Summary", expanded=True):
        summary = bv_solver.get_summary()
        st.text(summary)

    # Create visualization
    st.subheader("Visualization")

    # Display some key metrics first
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Max Potential",
            f"{results['max_potential']['max_value']:.4f}",
            help="Maximum absolute potential value in the domain"
        )

    with col2:
        field_val = results['field_at_point']['values']['field_magnitude']
        st.metric(
            "Field Magnitude",
            f"{field_val:.4f}",
            help="Electric field magnitude at evaluation point"
        )

    with col3:
        source_status = "✓ Inside" if results['source']['inside_domain'] else "✗ Outside"
        st.metric(
            "Source Status",
            source_status,
            help="Whether the source is inside the domain"
        )

    # === DISPLAY CACHED PLOTS (No recomputation!) ===

    # Plot 1: Domain Comparison
    st.markdown("### Domain Comparison")
    st.pyplot(fig1)

    # Plot 2: Potential Distribution
    st.markdown("### Electric Potential Distribution")
    st.pyplot(fig2)

    # Plot 3: Electric Field
    st.markdown("### Electric Field Distribution")
    st.pyplot(fig3)

    # === DOWNLOAD SECTION ===
    st.markdown("### 📥 Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Use cached MATLAB script
        st.download_button(
            label="📄 Download MATLAB Script",
            data=matlab_script,
            file_name="conformal_solver.m",
            mime="text/plain",
            help="Download complete MATLAB .m script with solution code",
            use_container_width=True
        )

    with col2:
        # Save potential plot as PNG (from cached figure)
        buf = io.BytesIO()
        fig2.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.download_button(
            label="🖼️ Download Potential Plot (PNG)",
            data=buf,
            file_name="potential_distribution.png",
            mime="image/png",
            help="Download high-resolution potential plot (300 DPI)",
            use_container_width=True
        )

    with col3:
        # Save field plot as PNG (from cached figure)
        buf_field = io.BytesIO()
        fig3.savefig(buf_field, format='png', dpi=300, bbox_inches='tight')
        buf_field.seek(0)

        st.download_button(
            label="🖼️ Download Field Plot (PNG)",
            data=buf_field,
            file_name="field_distribution.png",
            mime="image/png",
            help="Download high-resolution field plot (300 DPI)",
            use_container_width=True
        )

    # Optional: SVG exports
    with st.expander("🎨 Additional Export Options"):
        col_svg1, col_svg2 = st.columns(2)

        with col_svg1:
            # Save potential plot as SVG (from cached figure)
            buf_svg = io.BytesIO()
            fig2.savefig(buf_svg, format='svg', bbox_inches='tight')
            buf_svg.seek(0)

            st.download_button(
                label="📐 Potential Plot (SVG Vector)",
                data=buf_svg,
                file_name="potential_distribution.svg",
                mime="image/svg+xml",
                help="Vector format for scaling without quality loss",
                use_container_width=True
            )

        with col_svg2:
            # Save field plot as SVG (from cached figure)
            buf_svg_field = io.BytesIO()
            fig3.savefig(buf_svg_field, format='svg', bbox_inches='tight')
            buf_svg_field.seek(0)

            st.download_button(
                label="📐 Field Plot (SVG Vector)",
                data=buf_svg_field,
                file_name="field_distribution.svg",
                mime="image/svg+xml",
                help="Vector format for scaling without quality loss",
                use_container_width=True
            )

else:
    # Show welcome message if no results computed yet
    st.info("""
    👈 **Configure your problem in the sidebar**, then click **Compute Solution**.

    ### Quick Start Guide:
    1. **Select a geometry** (Circle, Rectangle, Square, or L-Shape)
    2. **Choose a conformal map** (try Identity first, then experiment with others)
    3. **Set source location** using the sliders (x₀, y₀)
    4. **Select boundary conditions** (Dirichlet sets V=constant on boundary)
    5. Click **Compute Solution** to see results!

    ### What This App Does:
    - Solves the 2D Laplace equation ∇²φ = -δ(r - r₀) using Green's functions
    - Applies conformal mapping to transform complex geometries
    - Visualizes electric potential and field distributions
    - Exports results to MATLAB scripts and high-resolution images
    """)

    # Display example images or equations
    st.markdown("### Theory")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Green's Function (2D Laplace)**

        For a point source at (x₀, y₀):

        ```
        G(x,y; x₀,y₀) = -ln(r) / (2π)

        where r = √[(x-x₀)² + (y-y₀)²]
        ```

        The potential φ is proportional to G.
        """)

    with col2:
        st.markdown("""
        **Conformal Mapping**

        Transforms complex domains to simpler ones:

        ```
        w = f(z)    where z = x + iy
        ```

        Examples:
        - Circle → Strip: w = -i·ln(z)
        - Möbius: w = (az+b)/(cz+d)
        - Joukowski: w = z + 1/z
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
**Conformal Field Solver v1.0**<br>
University Project<br>
Electromagnetic Boundary Value Problems
</small>
""", unsafe_allow_html=True)
