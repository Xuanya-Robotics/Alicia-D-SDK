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

from typing import List, Tuple, Union
import torch


# Refer to https://iquilezles.org/articles/distfunctions/
from typing import List, Tuple
import torch

class SimpleShapeSDF:
    def __init__(self, type: str, size: List[str]):
        self.type = type
        self.size = size

    def transform_points(self, points: torch.Tensor, base_trans: torch.Tensor) -> torch.Tensor:
        """
        Apply the base transformation to the input points.
        :param points: Tensor of shape (n, 3), representing points in 3D space.
        :param base_trans: Tensor of shape (1, 4, 4), representing the transformation matrix.
        :return: Transformed points, Tensor of shape (n, 3).
        """
        B = len(base_trans)  # Batch size
        N = len(points)  # Number of points

        # Compute the inverse of base_trans to transform points from base (global) to object coordinates
        base_trans_inv = torch.linalg.inv(base_trans)

        # Apply the inverse transformation: base_trans_inv is (4, 4), points is (N, 3)
        # We need to add a homogeneous coordinate (1) to the points to apply the transformation
        ones = torch.ones(points.shape[0], 1, device=points.device)
        points_homogeneous = torch.cat([points, ones], dim=1)  # Shape: (N, 4)
        points_homogeneous = points_homogeneous.resize(B, N, 4)
        # Apply the inverse transformation matrix: base_trans_inv @ points_homogeneous.T
        points_transformed = torch.matmul(base_trans_inv, points_homogeneous.permute(0, 2, 1)).permute(0, 2, 1)[:, :,
                             :3]  # Transform and drop the homogeneous coordinate

        # Now the points are in the object frame
        points = points_transformed

        # Return the transformed 3D points (dropping the homogeneous coordinate)
        return points

    def get_sdf_batch(self, points: torch.Tensor, base_trans: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the SDF and gradient for a batch of points.
        :param points: Tensor of shape (n, 3), representing points in 3D space.
        :param base_trans: Optional Tensor of shape (1, 4, 4), transformation matrix to apply to the points.
        :return: Tuple of (SDF values, gradients), where SDF values has shape (n,) and gradients has shape (n, 3).
        """
        if base_trans is not None:
            # Transform the points using the provided base transformation matrix
            points = self.transform_points(points, base_trans)

        if self.type == 'sphere':
            self.size = torch.tensor(self.size).to(points.device)
            return sphere_sdf_grad(points, self.size['r'])
        elif self.type == 'box':
            self.size = torch.tensor([float(i) for i in self.size]).to(points.device)
            return box_sdf_grad(points, self.size)
        else:
            raise ValueError(f"Unsupported shape type: {self.type}")

def sphere_sdf_grad(p: Union[torch.Tensor, List[float]], r: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SDF of a sphere

    >>> sphere_sdf_grad(torch.tensor([[2.0, 0.0, 0.0]]), 1.0)
    (tensor([1.]), tensor([[1., 0., 0.]]))

    :param p: point cloud
    :param r: radius
    :return: SDF and gradient
    """
    sdf = torch.norm(p, dim=-1) - r
    grad = torch.nn.functional.normalize(p, dim=-1)
    return sdf, grad


def box_sdf_grad(p: Union[torch.Tensor, List[List[float]]],
                 b: Union[torch.Tensor, List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the unsigned distance (SDF) and gradient for a 3D rectangular box (cuboid).

    :param p: Point cloud of shape [B, N, 3], where B is the batch size and N is the number of points.
    :param b: Box half-size of shape [3] or [B, 3] (can be specified per batch or shared across batches).
    :return: A tuple of SDF values (shape [B, N]) and gradients (shape [B, N, 3]).
    """
    b = b/2
    # Ensure tensor format for both p and b
    if isinstance(p, list):
        p = torch.tensor(p)
    if isinstance(b, list):
        b = torch.tensor(b)

    # Handle the case where box size `b` is either [3] or [B, 3]
    if b.dim() == 1:
        b = b.unsqueeze(0).expand(p.size(0), -1)  # Expand to [B, 3] if necessary (shared across batches)

    # Step 1: Compute q = p - clamp(p, -b, b)
    # This gives the distance along each axis to the nearest face of the box
    q = p - torch.clamp(p, -b.unsqueeze(1), b.unsqueeze(1))  # q shape: [B, N, 3]

    # Step 2: Compute the unsigned SDF as the Euclidean distance to the nearest surface
    sdf = torch.norm(q, dim=-1)  # Unsigned Euclidean distance to the surface (shape: [B, N])

    # Step 3: Compute the gradient
    # For points outside the box, normalize q to get the direction to the nearest surface
    grad = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)  # Gradient shape: [B, N, 3]

    return sdf, grad


def cylinder_sdf_grad(p: Union[torch.Tensor, List[float]], r: float, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SDF of a cylinder

    >>> cylinder_sdf_grad(torch.tensor([[2.0, 0.0, 0.0]]), 1.0, 1.0)
    (tensor([1.]), tensor([[1., 0., 0.]]))

    :param p: point cloud
    :param r: radius
    :param h: height
    :return: SDF and gradient
    """
    d = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - r
    sdf = torch.max(d, torch.abs(p[:, 1]) - h)
    grad = torch.zeros_like(p)
    grad[:, 0] = torch.where(d > 0, p[:, 0], 0)
    grad[:, 1] = torch.where(torch.abs(p[:, 1]) - h > 0, torch.sign(p[:, 1]), 0)
    grad[:, 2] = torch.where(d > 0, p[:, 2], 0)
    grad = torch.nn.functional.normalize(grad, dim=-1)
    return sdf, grad


def capsule_sdf_grad(p: Union[torch.Tensor, List[float]], r: float, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SDF of a capsule

    >>> capsule_sdf_grad(torch.tensor([[2.0, 0.0, 0.0]]), 1.0, 1.0)
    (tensor([1.]), tensor([[1., 0., 0.]]))

    :param p: point cloud
    :param r: radius
    :param h: height
    :return: SDF and gradient
    """
    d = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - r
    sdf = torch.max(d, torch.abs(p[:, 1]) - h)
    grad = torch.zeros_like(p)
    grad[:, 0] = torch.where(d > 0, p[:, 0], 0)
    grad[:, 1] = torch.where(torch.abs(p[:, 1]) - h > 0, torch.sign(p[:, 1]), 0)
    grad[:, 2] = torch.where(d > 0, p[:, 2], 0)
    grad = torch.nn.functional.normalize(grad, dim=-1)
    return sdf, grad


def torus_sdf_grad(p: Union[torch.Tensor, List[float]], r1: float, r2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SDF of a torus

    >>> torus_sdf_grad(torch.tensor([[2.0, 0.0, 0.0]]), 1.0, 0.5)
    (tensor([0.5000]), tensor([[1., 0., 0.]]))

    :param p: point cloud
    :param r1: external radius
    :param r2: internal radius
    :return: SDF and gradient
    """
    q = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - r1
    sdf = torch.sqrt(q ** 2 + p[:, 1] ** 2) - r2
    grad = torch.zeros_like(p)
    grad[:, 0] = torch.where(q > 0, p[:, 0], 0)
    grad[:, 1] = torch.where(sdf > 0, p[:, 1], 0)
    grad[:, 2] = torch.where(q > 0, p[:, 2], 0)
    grad = torch.nn.functional.normalize(grad, dim=-1)
    return sdf, grad


def ellipsoid_sdf_grad(p: Union[torch.Tensor, List[float]], r: Union[torch.Tensor, List[float]]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    SDF of an ellipsoid

    >>> ellipsoid_sdf_grad(torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([1.0, 1.0, 1.0]))
    (tensor([1.]), tensor([[1., 0., 0.]]))

    :param p: point cloud
    :param r: radius
    :return: SDF and gradient
    """
    q = p / r
    sdf = torch.norm(q, dim=-1) - 1.0
    grad = torch.nn.functional.normalize(q, dim=-1)
    return sdf, grad


import torch
import numpy as np


def sdf_sphere(points, center, radius):
    """
    Calculate the Signed Distance Function (SDF) for a sphere for a batch of points using PyTorch or NumPy.

    :param points: A batch of query points of shape (N, 3), where N is the batch size.
                   Can be a NumPy array or a PyTorch tensor.
    :param center: The center of the sphere (3D vector). Can be a NumPy array or a PyTorch tensor.
    :param radius: The radius of the sphere (float or tensor).
    :return: A tensor of shape (N,) with the signed distance from each point to the surface of the sphere.
    """
    np_flag = isinstance(points, np.ndarray)

    points = torch.as_tensor(points, device='cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU if available
    center = torch.as_tensor(center, device=points.device)
    radius = torch.as_tensor(radius, device=points.device)

    # Compute the signed distance
    distances = torch.norm(points - center, dim=-1) - radius

    if np_flag:
        return distances.cpu().numpy()  # Move back to CPU if necessary
    return distances


def sdf_box(points, center, half_size):
    """
    Calculate the Signed Distance Function (SDF) for a box for a batch of points using PyTorch or NumPy.

    :param points: A batch of query points of shape (N, 3), where N is the batch size.
                   Can be a NumPy array or a PyTorch tensor.
    :param center: The center of the box (3D vector). Can be a NumPy array or a PyTorch tensor.
    :param half_size: Half the size of the box in each dimension as a 3D tensor [half_x, half_y, half_z].
                      Can be a NumPy array or a PyTorch tensor.
    :return: A tensor of shape (N,) with the signed distance from each point to the surface of the box.
    """
    np_flag = isinstance(points, np.ndarray)

    points = torch.as_tensor(points, device='cuda' if torch.cuda.is_available() else 'cpu')
    center = torch.as_tensor(center, device=points.device)
    half_size = torch.as_tensor(half_size, device=points.device)

    q = torch.abs(points - center) - half_size
    distances = torch.norm(torch.maximum(q, torch.zeros_like(q)), dim=-1) + torch.minimum(torch.max(q, dim=-1).values,
                                                                                          torch.tensor(0.0,
                                                                                                       device=points.device))

    if np_flag:
        return distances.cpu().numpy()
    return distances


def sdf_cylinder(points, center, radius, height):
    """
    Calculate the Signed Distance Function (SDF) for a cylinder for a batch of points using PyTorch or NumPy.

    :param points: A batch of query points of shape (N, 3), where N is the batch size.
                   Can be a NumPy array or a PyTorch tensor.
    :param center: The center of the cylinder (3D vector). Can be a NumPy array or a PyTorch tensor.
    :param radius: The radius of the cylinder (float or tensor).
    :param height: The height of the cylinder (float or tensor).
    :return: A tensor of shape (N,) with the signed distance from each point to the surface of the cylinder.
    """
    np_flag = isinstance(points, np.ndarray)

    points = torch.as_tensor(points, device='cuda' if torch.cuda.is_available() else 'cpu')
    center = torch.as_tensor(center, device=points.device)
    radius = torch.as_tensor(radius, device=points.device)
    height = torch.as_tensor(height, device=points.device)

    d = torch.stack([
        torch.norm(points[..., :2] - center[:2], dim=-1) - radius,
        torch.abs(points[..., 2] - center[2]) - height / 2
    ], dim=-1)

    distances = torch.norm(torch.maximum(d, torch.zeros_like(d)), dim=-1) + torch.minimum(torch.max(d, dim=-1).values,
                                                                                          torch.tensor(0.0,
                                                                                                       device=points.device))

    if np_flag:
        return distances.cpu().numpy()
    return distances


def sdf_capsule(points, from_point, to_point, radius, height):
    """
    Calculate the Signed Distance Function (SDF) for a capsule for a batch of points using PyTorch or NumPy.

    :param points: A batch of query points of shape (N, 3), where N is the batch size.
                   Can be a NumPy array or a PyTorch tensor.
    :param from_point: The starting point of the capsule's line segment as a 3D vector.
                       Can be a NumPy array or a PyTorch tensor.
    :param to_point: The ending point of the capsule's line segment as a 3D vector.
                     Can be a NumPy array or a PyTorch tensor.
    :param radius: The radius of the capsule (float or tensor).
    :param height: The height (length) of the capsule's cylindrical part (float or tensor).
    :return: A tensor of shape (N,) with the signed distance from each point to the surface of the capsule.
    """
    np_flag = isinstance(points, np.ndarray)

    points = torch.as_tensor(points, device='cuda' if torch.cuda.is_available() else 'cpu')
    from_point = torch.as_tensor(from_point, device=points.device)
    to_point = torch.as_tensor(to_point, device=points.device)
    radius = torch.as_tensor(radius, device=points.device)
    height = torch.as_tensor(height, device=points.device)

    # Calculate the vector along the capsule's central line (from_point to to_point)
    capsule_axis = to_point - from_point

    # Normalize the axis to get the direction vector
    axis_length = torch.norm(capsule_axis, dim=-1)
    capsule_axis_normalized = capsule_axis / axis_length

    # Vector from the start of the capsule to the query points
    points_to_start = points - from_point

    # Project the points onto the capsule's axis
    t = torch.sum(points_to_start * capsule_axis_normalized, dim=-1, keepdim=True)

    # Clamp t to the range [0, axis_length] to handle the cylindrical part
    t_clamped = torch.clamp(t, 0.0, axis_length)

    # Find the closest points on the capsule's axis to the query points
    closest_points_on_axis = from_point + capsule_axis_normalized * t_clamped

    # Compute the distance from the query points to the closest points on the axis
    distance_to_axis = torch.norm(points - closest_points_on_axis, dim=-1)

    # The signed distance is the distance to the axis minus the capsule's radius
    distances = distance_to_axis - radius

    if np_flag:
        return distances.cpu().numpy()
    return distances
