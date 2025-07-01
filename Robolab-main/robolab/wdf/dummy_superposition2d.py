#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.widgets import CheckButtons


# Create a 2D grid
def create_grid(resolution, grid_size):
    x = torch.linspace(-grid_size, grid_size, resolution, requires_grad=True)  # Enable requires_grad
    y = torch.linspace(-grid_size, grid_size, resolution, requires_grad=True)  # Enable requires_grad
    X, Y = torch.meshgrid(x, y, indexing='ij')  # Note that PyTorch's meshgrid defaults to 'ij' indexing
    X.requires_grad_()  # Ensure X can compute gradients
    Y.requires_grad_()  # Ensure Y can compute gradients
    return X, Y


# Calculate the SDF (signed distance function) for a point (distance field to a circle)
def sdf_circle(X, Y, cx, cy, radius):
    return torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius


# Calculate the combined SDF by accumulating distance fields from multiple circles
def sdf_combined(X, Y, circles):
    field = torch.full(X.shape, float('inf'), requires_grad=True)  # Enable requires_grad
    for (cx, cy, radius) in circles:
        field = torch.minimum(field, sdf_circle(X, Y, cx, cy, radius))
    return field


# Plot the distance field and arrows
def plot_sdf(ax, X, Y, field, show_arrows, show_color, colorbar=None):
    ax.clear()
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)

    if show_color:
        # Plot the distance field as a contour plot, convert PyTorch tensors to NumPy arrays
        contour = ax.contourf(X.detach().numpy(), Y.detach().numpy(), field.detach().numpy(), levels=50,
                              cmap=cm.viridis)

        # Ensure the colorbar is only added once
        if colorbar is None:
            colorbar = plt.colorbar(contour, ax=ax)

    if show_arrows:
        # Compute the gradient using torch.autograd.grad
        grad_x, grad_y = torch.autograd.grad(field.sum(), [X, Y], create_graph=True)

        # For areas outside the object, the gradient direction should point towards the boundary
        inside_mask = field > 0  # Select the area outside the object
        grad_x[inside_mask] = -grad_x[inside_mask]  # Reverse the gradient in areas outside the object
        grad_y[inside_mask] = -grad_y[inside_mask]  # Reverse the gradient in areas outside the object

        # Plot the arrows, convert PyTorch tensors to NumPy arrays
        ax.quiver(
            X[::3, ::3].detach().numpy(),
            Y[::3, ::3].detach().numpy(),
            grad_x[::3, ::3].detach().numpy(),
            grad_y[::3, ::3].detach().numpy(),
            color='black',
            scale=10,  # Adjust scale to reduce arrow size
            scale_units='xy',  # Use 'xy' units for consistent scaling
            width=0.002  # Reduce arrow width
        )

    # Draw all circles
    for (cx, cy, radius) in circles:
        circle = Circle((cx, cy), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)

    return colorbar


# Mouse click event handler
def on_click(event):
    if event.inaxes:
        # Get the coordinates of the click
        cx, cy = event.xdata, event.ydata
        radius = 1  # Fixed radius
        circles.append((cx, cy, radius))  # Add new circle
        field = sdf_combined(X, Y, circles)  # Update the SDF

        # Redraw, keeping the colorbar unchanged
        plot_sdf(ax, X, Y, field, show_arrows, show_color, colorbar)
        plt.draw()


# Checkbox event handler
def on_checkbox_change(label):
    global show_arrows, show_color
    if label == 'Show Arrows':
        show_arrows = not show_arrows
    elif label == 'Show Color':
        show_color = not show_color

    # Update the plot
    field = sdf_combined(X, Y, circles)
    plot_sdf(ax, X, Y, field, show_arrows, show_color, colorbar)
    plt.draw()


if __name__ == '__main__':
    # Initialize data
    resolution = 100
    grid_size = 5
    X, Y = create_grid(resolution, grid_size)
    circles = [(0, 0, 1)]  # Initially, there is a circle with center (0, 0) and radius 1

    # Create the figure and UI
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25)  # Adjust figure size to accommodate checkboxes

    # Initial display of arrows and color
    show_arrows = True
    show_color = True

    # Initial SDF field and display
    field = sdf_combined(X, Y, circles)
    colorbar = plot_sdf(ax, X, Y, field, show_arrows, show_color)

    # Add checkboxes
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])  # Adjust position and size
    labels = ['Show Arrows', 'Show Color']
    visibility = [show_arrows, show_color]
    check = CheckButtons(rax, labels, visibility)

    # Connect checkbox event
    check.on_clicked(on_checkbox_change)

    # Connect mouse click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
