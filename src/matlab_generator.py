"""
MATLAB Script Generator

Generates MATLAB .m scripts that reproduce the solution computed
by the Conformal Field Solver.
"""

import numpy as np


def generate_matlab_script(results, geometry_name, conformal_map, source_x, source_y,
                          source_strength, bc_type, bc_value, nx, ny):
    """
    Generate a MATLAB script that reproduces the solution.

    Parameters:
    -----------
    results : dict
        Results dictionary from the solver
    geometry_name : str
        Name of the geometry
    conformal_map : str
        Name of the conformal mapping
    source_x, source_y : float
        Source coordinates
    source_strength : float
        Source strength
    bc_type : str
        Boundary condition type
    bc_value : float
        Boundary condition value
    nx, ny : int
        Grid resolution

    Returns:
    --------
    matlab_code : str
        Complete MATLAB script
    """

    # Extract bounds from results
    X = results['X']
    Y = results['Y']
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)

    # Map geometry names to MATLAB comments
    geometry_descriptions = {
        'circle': 'Circle (radius 1, centered at origin)',
        'rectangle': 'Rectangle',
        'square': 'Square',
        'lshape': 'L-shaped domain',
        'annulus': 'Annulus (circular ring)',
        'semiplane': 'Semi-infinite plane (y ≥ 0)',
        'wedge': 'Wedge/Sector'
    }

    # Map conformal map names to MATLAB code
    conformal_map_code = {
        'identity': 'w = z;  % Identity (no transformation)',
        'circle_to_strip': 'w = -1i * log(z);  % Circle to strip',
        'mobius': 'w = z + 1;  % Möbius (translation by 1)',
        'inversion': 'w = 1 ./ z;  % Inversion',
        'joukowski': 'w = z + 1 ./ z;  % Joukowski transformation',
        'square': 'w = z.^2;  % Square mapping',
        'sine': 'w = sin(z);  % Sine mapping',
        'custom': 'w = z;  % Custom function (edit as needed)'
    }

    matlab_code = f"""% Conformal Field Solver - MATLAB Script
% Generated automatically
%
% Problem: Electromagnetic Boundary Value Problem
% Geometry: {geometry_descriptions.get(geometry_name, geometry_name)}
% Conformal Map: {conformal_map}
%
% Solves the 2D Laplace equation using Green's functions
% and conformal mapping.

clear all; close all; clc;

%% Problem Parameters
fprintf('\\n=== Conformal Field Solver ===\\n\\n');

% Source location and strength
x0 = {source_x};
y0 = {source_y};
source_strength = {source_strength};

fprintf('Source location: (%.3f, %.3f)\\n', x0, y0);
fprintf('Source strength: %.3f\\n', source_strength);

% Boundary condition
bc_type = '{bc_type}';
bc_value = {bc_value};

% Grid resolution
nx = {nx};
ny = {ny};

%% Define Grid
xmin = {xmin:.4f};
xmax = {xmax:.4f};
ymin = {ymin:.4f};
ymax = {ymax:.4f};

x = linspace(xmin, xmax, nx);
y = linspace(ymin, ymax, ny);
[X, Y] = meshgrid(x, y);

%% Green's Function for 2D Laplace Equation
% G(x,y; x0,y0) = -ln(r) / (2*pi)
% where r = sqrt((x-x0)^2 + (y-y0)^2)

function G = greens_laplace_2d(X, Y, x0, y0)
    r = sqrt((X - x0).^2 + (Y - y0).^2);
    r = max(r, 1e-10);  % Avoid singularity
    G = -log(r) / (2 * pi);
end

%% Compute Potential
phi = source_strength * greens_laplace_2d(X, Y, x0, y0);

%% Apply Conformal Mapping
z = X + 1i*Y;
{conformal_map_code.get(conformal_map, 'w = z;  % Identity')}

U = real(w);
V = imag(w);

% Transform source location
z_source = x0 + 1i*y0;
{conformal_map_code.get(conformal_map, 'w_source = z_source;').replace('w =', 'w_source =')}
x0_transformed = real(w_source);
y0_transformed = imag(w_source);

fprintf('Transformed source: (%.3f, %.3f)\\n', x0_transformed, y0_transformed);

% Potential in transformed domain
phi_transformed = source_strength * greens_laplace_2d(U, V, x0_transformed, y0_transformed);

%% Compute Electric Field (E = -grad(phi))
[phi_y, phi_x] = gradient(phi, y(2)-y(1), x(2)-x(1));
Ex = -phi_x;
Ey = -phi_y;
E_magnitude = sqrt(Ex.^2 + Ey.^2);

%% Find Maximum Potential
[max_phi, idx] = max(abs(phi(:)));
[row, col] = ind2sub(size(phi), idx);
max_x = X(row, col);
max_y = Y(row, col);

fprintf('Maximum potential: %.6f at (%.3f, %.3f)\\n', max_phi, max_x, max_y);

%% Plotting

% Figure 1: Original and Transformed Domains
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
plot(x0, y0, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
hold on;
title('Original Domain', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('x'); ylabel('y');
grid on; axis equal;
legend('Source');

subplot(1, 2, 2);
plot(x0_transformed, y0_transformed, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
hold on;
title('Transformed Domain', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('u (Re w)'); ylabel('v (Im w)');
grid on; axis equal;
legend('Source (transformed)');

% Figure 2: Potential Distribution
figure('Position', [100, 100, 1000, 800]);
contourf(X, Y, phi, 20, 'LineStyle', 'none');
colorbar('Label', 'Potential \\phi');
colormap('jet');
hold on;
contour(X, Y, phi, 20, 'k', 'LineWidth', 0.5, 'LineStyle', '-');
plot(x0, y0, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
plot(max_x, max_y, 'wo', 'MarkerSize', 10, 'MarkerFaceColor', 'w', ...
     'MarkerEdgeColor', 'k', 'LineWidth', 2);
title('Electric Potential \\phi(x,y)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('x'); ylabel('y');
legend('Source', 'Max Potential', 'Location', 'best');
grid on; axis equal;

% Figure 3: Electric Field
figure('Position', [100, 100, 1000, 800]);
contourf(X, Y, E_magnitude, 20, 'LineStyle', 'none');
colorbar('Label', 'Field Magnitude |E|');
colormap('hot');
hold on;

% Subsample vectors for clarity
step = max(1, floor(nx/20));
quiver(X(1:step:end, 1:step:end), Y(1:step:end, 1:step:end), ...
       Ex(1:step:end, 1:step:end), Ey(1:step:end, 1:step:end), ...
       'b', 'LineWidth', 1.5, 'AutoScale', 'on');

plot(x0, y0, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
title('Electric Field E = -\\nabla\\phi', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('x'); ylabel('y');
legend('Source', 'Location', 'best');
grid on; axis equal;

fprintf('\\n=== Computation Complete ===\\n');
fprintf('Figures generated successfully.\\n\\n');

% End of script
"""

    return matlab_code


