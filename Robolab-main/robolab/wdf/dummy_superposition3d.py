# Copyright (c) 2024, Junjia LIU, jjliu@mae.cuhk.,edu.hk
#
# This software is licensed under the GNU General Public License version 3 (GPL-3.0),
# with the following additional restriction:
#
# Any commercial use of this software, including but not limited to selling,
# sublicensing, or incorporating the software into a commercial product or service,
# requires the user to share a percentage of the profits generated from such use
# with the copyright holder. The percentage of profits to be shared shall be 50%
# of the net revenue generated from the software or any products or services that
# incorporate the software.
#
# For more details, see the LICENSE file.

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons
from matplotlib import cm


# Create a 3D grid
def create_grid(resolution, grid_size):
    x = torch.linspace(-grid_size, grid_size, resolution, requires_grad=True)
    y = torch.linspace(-grid_size, grid_size, resolution, requires_grad=True)
    z = torch.linspace(-grid_size, grid_size, resolution, requires_grad=True)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    X.requires_grad_()
    Y.requires_grad_()
    Z.requires_grad_()
    return X, Y, Z


# Calculate the SDF for a sphere in 3D space
def sdf_sphere(X, Y, Z, cx, cy, cz, radius):
    return torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) - radius


# Combine the SDF for multiple spheres
def sdf_combined(X, Y, Z, spheres):
    field = torch.full(X.shape, float('inf'), requires_grad=True)
    for (cx, cy, cz, radius) in spheres:
        field = torch.minimum(field, sdf_sphere(X, Y, Z, cx, cy, cz, radius))
    return field


# Plot the distance field and arrows in 3D
def plot_sdf(ax, X, Y, Z, field, show_arrows, show_color):
    ax.clear()
    ax.set_xlim([-grid_size, grid_size])
    ax.set_ylim([-grid_size, grid_size])
    ax.set_zlim([-grid_size, grid_size])

    # Convert to NumPy arrays for plotting
    X_np = X.detach().numpy()
    Y_np = Y.detach().numpy()
    Z_np = Z.detach().numpy()

    if show_color:
        # Plot each sphere as a surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for (cx, cy, cz, radius) in spheres:
            x_sphere = cx + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = cy + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot the surface of the sphere
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.6, rstride=5, cstride=5)

    if show_arrows:
        # Compute the gradient using torch.autograd.grad
        grad_x, grad_y, grad_z = torch.autograd.grad(field.sum(), [X, Y, Z], create_graph=True)

        # Sample the grid to reduce the number of arrows
        stride = 6  # Increase stride to reduce the number of markers
        X_sampled = X_np[::stride, ::stride, ::stride]
        Y_sampled = Y_np[::stride, ::stride, ::stride]
        Z_sampled = Z_np[::stride, ::stride, ::stride]
        grad_x_sampled = -grad_x[::stride, ::stride, ::stride].detach().numpy()
        grad_y_sampled = -grad_y[::stride, ::stride, ::stride].detach().numpy()
        grad_z_sampled = -grad_z[::stride, ::stride, ::stride].detach().numpy()

        # Compute the distance for each sampled point (SDF values)
        field_sampled = field[::stride, ::stride, ::stride].detach().numpy()
        field_sampled_normalized = (field_sampled - field_sampled.min()) / (
                    field_sampled.max() - field_sampled.min())  # Normalize distances

        # Get colors from the normalized SDF values using a colormap
        cmap = plt.get_cmap('viridis')
        colors = cmap(field_sampled_normalized.flatten())  # Use the normalized distance to get colors

        # Plot the arrows with very short shafts (almost only heads)
        ax.quiver(
            X_sampled,
            Y_sampled,
            Z_sampled,
            grad_x_sampled,
            grad_y_sampled,
            grad_z_sampled,
            length=0.5,  # Make the arrows very short
            color=colors,  # Use the distance-mapped colors
            linewidth=1.5,
            normalize=True
        )


# Function to add a new sphere based on user input
def add_sphere():
    while True:
        print("\nEnter the center coordinates (cx, cy, cz) and the radius of the sphere.")
        try:
            cx = float(input("Enter cx: "))
            cy = float(input("Enter cy: "))
            cz = float(input("Enter cz: "))
            radius = float(input("Enter radius: "))
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            continue

        # Add the new sphere to the list
        spheres.append((cx, cy, cz, radius))
        field = sdf_combined(X, Y, Z, spheres)  # Update the SDF

        # Redraw the plot
        plot_sdf(ax, X, Y, Z, field, show_arrows, show_color)
        plt.draw()  # Update the plot dynamically

        # Ask if the user wants to add another sphere
        another = input("\nDo you want to add another sphere? (yes/no): ").strip().lower()
        if another == 'no':
            break


# Checkbox event handler
def on_checkbox_change(label):
    global show_arrows, show_color
    if label == 'Show Arrows':
        show_arrows = not show_arrows
    elif label == 'Show Color':
        show_color = not show_color

    # Update the plot
    field = sdf_combined(X, Y, Z, spheres)
    plot_sdf(ax, X, Y, Z, field, show_arrows, show_color)
    plt.draw()  # Redraw the plot when checkboxes are changed


if __name__ == '__main__':
    # Set the backend explicitly to avoid backend issues
    plt.switch_backend('TkAgg')  # Ensure we're using a GUI backend that supports windowing

    # Initialize data
    resolution = 50  # 3D grid resolution
    grid_size = 5  # Size of the 3D grid
    X, Y, Z = create_grid(resolution, grid_size)
    spheres = [(0, 0, 0, 1)]  # Initially, there is one sphere with center (0, 0, 0) and radius 1

    # Create the figure and UI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.25)

    # Enable interactive mode
    plt.ion()  # Turn on interactive mode

    # Initial display of arrows and color
    show_arrows = True
    show_color = True

    # Initial SDF field and display
    field = sdf_combined(X, Y, Z, spheres)
    plot_sdf(ax, X, Y, Z, field, show_arrows, show_color)

    # Add checkboxes for controlling options
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    labels = ['Show Arrows', 'Show Color']
    visibility = [show_arrows, show_color]
    check = CheckButtons(rax, labels, visibility)

    # Connect checkbox event to handler
    check.on_clicked(on_checkbox_change)

    # Show the plot window in non-blocking mode
    plt.show(block=False)

    # Start the terminal input loop for adding spheres
    add_sphere()  # Call the function to start the input process

    # Keep the window open until the user manually closes it
    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Ensure the plot stays open for user interaction
