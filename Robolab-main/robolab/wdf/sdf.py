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

import os
import numpy as np
import torch
import trimesh
import skimage
import mesh_to_sdf
import torch.nn as nn
import torch.optim as optim
from robolab.formatter.urdf_parser.urdf import URDF
from robolab.formatter.mjcf_parser.mjcf import MJCF
from robolab.wdf.simple_shape_sdf import SimpleShapeSDF
from tqdm import tqdm


class SDFBase:
    def __init__(self, args):
        """
        Use Bernstein Polynomial to represent the SDF of a single mesh.
        """
        self.args = args
        self.domain_min = args.domainMin
        self.domain_max = args.domainMax
        self.device = args.device
        self.simple_shape = False
        if args.assetFile.startswith("box") or args.assetFile.startswith("sphere"):
            self.simple_shape = True
            self.shape_type = args.assetFile.split("_")[0]
            self.shape_size = args.assetFile.split("_")[1:]
            self.simple_sdf = SimpleShapeSDF(self.shape_type, self.shape_size)

        if not self.simple_shape:
            self.asset_path = os.path.join(args.assetRoot, args.assetFile)

            self.mesh_path, self.scale = self._extract_stl_from_asset()
            self.wdf_dir = os.path.join(os.path.dirname(self.asset_path), "sdf")
            self.save_mesh_dict = args.saveMeshDict

            # Load the mesh
            assert os.path.exists(self.mesh_path), "Mesh file does not exist!"
            self.mesh = trimesh.load(self.mesh_path)
            self.mesh = self.mesh.apply_scale(self.scale)

    def _extract_stl_from_asset(self):
        """
        Parse the MJCF file and extract the STL path of the first mesh.
        """
        assert os.path.exists(self.asset_path), f"ASSET file does not exist at path: {self.asset_path}"

        scale = [1, 1, 1]
        if self.asset_path.endswith('.stl'):
            print(f"Extracted STL path from ASSET: {self.asset_path}")
            return self.asset_path
        elif self.asset_path.endswith('.xml'):
            # Parse the MJCF file
            mjcf_model = MJCF(self.asset_path)
            assert len(mjcf_model.link_mesh_map) == 1, \
                "Only single link MJCFs are supported for SDF, for a MJCF with multiple links, please use RDF."

            mesh = mjcf_model.link_mesh_map[list(mjcf_model.link_mesh_map.keys())[0]]
            assert len(mesh) == 1, "Only single mesh elements are supported for SDF."
            relative_stl_path = mesh[list(mesh.keys())[0]]["params"]["mesh_path"]
            scale = mesh[list(mesh.keys())[0]]["params"]["scale"]
            stl_path = relative_stl_path

            print(f"Extracted STL path from MJCF: {stl_path}")
        elif self.asset_path.endswith('.urdf'):
            urdf_model = URDF.from_xml_file(self.asset_path)
            assert len(urdf_model.links) == 1, \
                "Only single link URDFs are supported for SDF, for a URDF with multiple links, please use RDF."

            relative_stl_path = urdf_model.links[0].collision.geometry.filename
            assert relative_stl_path.endswith('.stl'), "Only STL files are supported for SDF."
            stl_path = os.path.join(os.path.dirname(self.asset_path), relative_stl_path)

            print(f"Extracted STL path from URDF: {stl_path}")
        else:
            raise ValueError(f"Unsupported file format for SDF: {self.asset_path}")
        return stl_path, scale

    def train(self):
        raise NotImplementedError("Train method should be implemented in subclasses.")

    def sample_sdf_points(self):
        """
        Sample SDF points for the mesh and save them for training.
        """
        print(f"Sampling points for mesh at {self.mesh_path}...")

        # Scale the mesh to unit sphere for SDF calculation
        center = self.mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(self.mesh.vertices - center, axis=1))
        mesh = mesh_to_sdf.scale_to_unit_sphere(self.mesh)

        # Sample points near the surface (as in DeepSDF)
        near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            number_of_points=500000,
            surface_point_method='scan',
            sign_method='normal',
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=100,
            return_gradients=False
        )

        # Random points in the bounding box [-1, 1] for SDF sampling
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_sdf = mesh_to_sdf.mesh_to_sdf(mesh, random_points)

        # Save the sampled SDF points
        data = {
            'near_points': near_points,
            'near_sdf': near_sdf,
            'random_points': random_points,
            'random_sdf': random_sdf,
            'center': center,
            'scale': scale
        }
        os.makedirs(self.wdf_dir, exist_ok=True)
        np.save(os.path.join(self.wdf_dir, 'sampled_sdf_data.npy'), data)
        print(f"Sampled SDF points saved at {self.wdf_dir}/sampled_sdf_data.npy")

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        raise NotImplementedError("Create surface mesh method should be implemented in subclasses.")

    def get_sdf_batch(self, points, model, base_trans=None, use_derivative=False):
        """
        Get SDF values for a batch of points, optionally applying a base transformation.
        """
        if base_trans is None:
            B = 1
        else:
            B = len(base_trans)  # Batch size
        N = len(points)  # Number of points

        points = torch.tensor(points, dtype=torch.float32, device=self.device)

        if base_trans is not None:
            # Ensure base_trans is on the same device and in the correct shape
            base_trans = base_trans.to(self.device).float()

            # Compute the inverse of base_trans to transform points from base (global) to object coordinates
            base_trans_inv = torch.linalg.inv(base_trans)

            # Apply the inverse transformation: base_trans_inv is (4, 4), points is (N, 3)
            # We need to add a homogeneous coordinate (1) to the points to apply the transformation
            ones = torch.ones(points.shape[0], 1, device=self.device)  # Shape: (N, 1)
            points_homogeneous = torch.cat([points, ones], dim=1)  # Shape: (N, 4)
            points_homogeneous = points_homogeneous.resize(B, N, 4)
            # Apply the inverse transformation matrix: base_trans_inv @ points_homogeneous.T
            points_transformed = torch.matmul(base_trans_inv, points_homogeneous.permute(0, 2, 1)).permute(0, 2, 1)[:,
                                 :, :3]  # Transform and drop the homogeneous coordinate

            # Now the points are in the object frame
            points = points_transformed
            
        return B, N, points


class SDFBP(SDFBase):
    def __init__(self, args):
        """
        Use Bernstein Polynomial to represent the SDF of a single mesh.
        """
        super().__init__(args)
        self.n_func = args.numFuncs
        

    def _binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def _build_bernstein_t(self, t, use_derivative=False):
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self._binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def _build_basis_function_from_points(self, points, use_derivative=False):
        N = len(points)
        points = ((points - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self._build_bernstein_t(points, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func ** 3)
        if not use_derivative:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_y = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func ** 3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def train(self):
        """
        Train the SDF for the loaded mesh using sampled points.
        """
        # Load sampled SDF points
        if not os.path.exists(os.path.join(self.wdf_dir, 'sampled_sdf_data.npy')):
            self.sample_sdf_points()
        sampled_point_data = np.load(os.path.join(self.wdf_dir, 'sampled_sdf_data.npy'), allow_pickle=True).item()

        center = sampled_point_data['center']
        scale = sampled_point_data['scale']
        point_near_data = sampled_point_data['near_points']
        sdf_near_data = sampled_point_data['near_sdf']
        point_random_data = sampled_point_data['random_points']
        sdf_random_data = sampled_point_data['random_sdf']
        sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]

        wb = torch.zeros(self.n_func ** 3).float().to(self.device)
        batch_size = (torch.eye(self.n_func ** 3) / 1e-4).float().to(self.device)

        # Training loop using Bernstein polynomials
        for iter in range(self.args.trainEpochs):
            choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
            p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(
                self.device), torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)

            choice_random = np.random.choice(len(point_random_data), 256, replace=False)
            p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(
                self.device), torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)

            p = torch.cat([p_near, p_random], dim=0)
            sdf = torch.cat([sdf_near, sdf_random], dim=0)
            phi_xyz, _ = self._build_basis_function_from_points(p.float().to(self.device), use_derivative=False)

            K = torch.matmul(batch_size, phi_xyz.T).matmul(torch.linalg.inv(
                (torch.eye(len(p)).float().to(self.device) + torch.matmul(torch.matmul(phi_xyz, batch_size),
                                                                          phi_xyz.T))))
            batch_size -= torch.matmul(K, phi_xyz).matmul(batch_size)
            delta_wb = torch.matmul(K, (sdf - torch.matmul(phi_xyz, wb)).squeeze())
            wb += delta_wb

        # Save the trained SDF weights
        trained_model = {
            'weights': wb,
            'offset': torch.from_numpy(center),
            'scale': scale,
        }
        if self.save_mesh_dict:
            os.makedirs(os.path.join(self.wdf_dir, 'BP'), exist_ok=True)
            torch.save(trained_model, os.path.join(self.wdf_dir, 'BP', f'BP_{self.args.numFuncs}.pt'))
            print(f"Trained model saved at {self.wdf_dir}/BP/BP_{self.args.numFuncs}.pt")

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        """
        Create a surface mesh from the trained SDF model using marching cubes.
        """
        weights = model['weights'].to(self.device)
        domain = torch.linspace(self.domain_min, self.domain_max, nbData).to(self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
        grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
        p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

        # Split data to handle memory limits
        p_split = torch.split(p, 10000, dim=0)
        d = []
        for p_s in p_split:
            phi_p, _ = self._build_basis_function_from_points(p_s, use_derivative=False)
            d_s = torch.matmul(phi_p, weights)
            d.append(d_s)
        d = torch.cat(d, dim=0)

        verts, faces, normals, values = skimage.measure.marching_cubes(
            d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0,
            spacing=np.array([(self.domain_max - self.domain_min) / nbData] * 3)
        )
        verts = verts - [1, 1, 1]

        # Create trimesh object and save/visualize if needed
        rec_mesh = trimesh.Trimesh(verts, faces)
        if vis:
            rec_mesh.show()
        if save_mesh_name is not None:
            save_path = os.path.join(self.wdf_dir, "output_meshes")
            os.makedirs(save_path, exist_ok=True)
            trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path, f"{save_mesh_name}.stl"))

    def get_sdf_batch(self, points, model, base_trans=None, use_derivative=False):
        """
        Get SDF values for a batch of points, optionally applying a base transformation.
        """
        B, N, points = super().get_sdf_batch(points, model, base_trans, use_derivative)

        if self.simple_shape:
            sdf, grad = self.simple_sdf.get_sdf_batch(points)
            gradient = torch.nn.functional.normalize(grad, dim=-1).float()
            if base_trans is not None:
                rotation_matrix = base_trans[:, :3, :3]  # (3, 3)
                gradient_base_frame = torch.einsum('ijk,ikl->ijl', rotation_matrix, gradient.transpose(1, 2)).transpose(
                    1, 2).reshape(B, N, 3)
                gradient = gradient_base_frame
            return sdf, gradient

        offset = model['offset'].to(self.device)
        scale = model['scale']
        weights = model['weights'].to(self.device)

        points_scaled = (points - offset) / scale  # Adjust points using offset and scale
        points_bounds = torch.where(points_scaled > 1.0 - 1e-2, 1.0 - 1e-2, points_scaled)
        points_bounds = torch.where(points_bounds < -1.0 + 1e-2, -1.0 + 1e-2, points_bounds)
        res_points = points_scaled - points_bounds

        if not use_derivative:
            phi, _ = self._build_basis_function_from_points(points_scaled.reshape(-1, 3), use_derivative)

            # sdf
            sdf = torch.matmul(phi, weights).reshape(B, N)
            sdf = sdf + res_points.norm(dim=-1)
            sdf = sdf * scale
            return sdf, None
        else:
            phi, dphi = self._build_basis_function_from_points(points_scaled.reshape(B * N, 3), use_derivative)

            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B * N, -1, 4)

            output = torch.einsum('jkl,k->jl', phi_cat, weights).reshape(B, N, 4)

            sdf = output[:, :, 0]
            sdf = sdf + res_points.norm(dim=-1)
            sdf = sdf * scale
            gradient = output[:, :, 1:]
            gradient = res_points + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()

            # Gradient transformation
            if base_trans is not None:
                # 从 base_trans 中提取旋转部分 (3x3)
                rotation_matrix = base_trans[:, :3, :3]  # (3, 3)

                # 将梯度从物体坐标系转换到基坐标系
                gradient_base_frame = torch.einsum('ijk,ikl->ijl', rotation_matrix, gradient.transpose(1, 2)).transpose(
                    1, 2).reshape( B, N, 3)
                gradient = gradient_base_frame
            return sdf, gradient


class SDFNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1):
        super(SDFNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh() # Consider using tanh for SDF output if values are bounded

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for SDF value, or tanh if bounded
        return x

class SDFNN(SDFBase):
    def __init__(self, args):
        """
        Use Bernstein Polynomial to represent the SDF of a single mesh.
        """
        super().__init__(args)
        self.hidden_dim = args.hiddenDim if hasattr(args, 'hiddenDim') else 256
        self.learning_rate = args.learningRate if hasattr(args, 'learningRate') else 1e-4
        self.nn_batch_size = args.nnBatchSize if hasattr(args, 'nnBatchSize') else 2048
    

    def train(self):
        # Load sampled SDF points
        if not os.path.exists(os.path.join(self.wdf_dir, 'sampled_sdf_data.npy')):
            self.sample_sdf_points()
        sampled_point_data = np.load(os.path.join(self.wdf_dir, 'sampled_sdf_data.npy'), allow_pickle=True).item()

        sdf_net = SDFNet(input_dim=3, hidden_dim=self.hidden_dim, output_dim=1).to(self.device)
        optimizer = optim.Adam(sdf_net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()  # L1Loss might also be good for SDF
            
        center = sampled_point_data['center']
        scale = sampled_point_data['scale']
        point_near_data = torch.from_numpy(sampled_point_data['near_points']).float().to(self.device)
        sdf_near_data = torch.from_numpy(sampled_point_data['near_sdf']).float().to(self.device).unsqueeze(-1)
        point_random_data = torch.from_numpy(sampled_point_data['random_points']).float().to(self.device)
        sdf_random_data = torch.from_numpy(sampled_point_data['random_sdf']).float().to(self.device).unsqueeze(-1)
        sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]

        all_points = torch.cat([point_near_data, point_random_data], dim=0)
        all_sdf = torch.cat([sdf_near_data, sdf_random_data], dim=0)

        epoch_pbar = tqdm(range(self.args.trainEpochs))
        for epoch in epoch_pbar:
            epoch_loss = 0
            optimizer.zero_grad()
            batch_points, batch_sdf = all_points, all_sdf

            pred_sdf = sdf_net(batch_points)
            loss = loss_fn(pred_sdf, batch_sdf)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            num_batches = (all_points.size(0) + self.nn_batch_size - 1) // self.nn_batch_size if all_points.size(
                0) > 0 else 1
            avg_loss = epoch_loss / num_batches * 1000
            epoch_pbar.set_postfix({'Loss': f"{avg_loss:.2f}"})

        # Save the trained SDF weights
        trained_model = {
            'model_state_dict': sdf_net.state_dict(),
            'offset': center,
            'scale': scale,
        }
        if self.save_mesh_dict:
            os.makedirs(os.path.join(self.wdf_dir, 'NN'), exist_ok=True)
            torch.save(trained_model, os.path.join(self.wdf_dir, 'NN', f'NN_h{self.hidden_dim}_e{self.args.trainEpochs}.pt'))
            print(f"Trained model saved at {self.wdf_dir}/NN/NN_h{self.hidden_dim}_e{self.args.trainEpochs}.pt")

    def get_model_dict(self, model):
        self.offset = torch.tensor(model['offset'], dtype=torch.float32).to(self.device)
        self.scale = torch.tensor(model['scale'], dtype=torch.float32).to(self.device)
        self.sdf_net = SDFNet(input_dim=3, hidden_dim=model.get('hidden_dim', self.hidden_dim),
                              output_dim=1).to(self.device)
        self.sdf_net.load_state_dict(model['model_state_dict'])
        self.sdf_net.eval()
        
    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        """
        Create a surface mesh from the trained SDF model using marching cubes.
        """
        if not hasattr(self, 'sdf_net'):
            self.get_model_dict(model)
        
        domain = torch.linspace(self.domain_min, self.domain_max, nbData).to(self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
        grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
        p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

        d = []
        with torch.no_grad():
            p_split = torch.split(p, 10000, dim=0)  # Batch processing for memory
            for p_s in p_split:
                sdf_values = self.sdf_net(p_s)
                d.append(sdf_values)
        d = torch.cat(d, dim=0)

        verts, faces, normals, values = skimage.measure.marching_cubes(
            d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0,
            spacing=np.array([(self.domain_max - self.domain_min) / nbData] * 3)
        )
        verts = verts - [1, 1, 1]

        # Create trimesh object and save/visualize if needed
        rec_mesh = trimesh.Trimesh(verts, faces)
        if vis:
            rec_mesh.show()
        if save_mesh_name is not None:
            save_path = os.path.join(self.wdf_dir, "output_meshes")
            os.makedirs(save_path, exist_ok=True)
            trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path, f"{save_mesh_name}.stl"))


                    
    def get_sdf_batch(self, points, model, base_trans=None, use_derivative=False):
        """
        Get SDF values for a batch of points, optionally applying a base transformation.
        """
        B, N, points = super().get_sdf_batch(points, model, base_trans, use_derivative)

        if self.simple_shape:
            sdf, grad = self.simple_sdf.get_sdf_batch(points)
            gradient = torch.nn.functional.normalize(grad, dim=-1).float()
            if base_trans is not None:
                rotation_matrix = base_trans[:, :3, :3]  # (3, 3)
                gradient_base_frame = torch.einsum('ijk,ikl->ijl', rotation_matrix, gradient.transpose(1, 2)).transpose(
                    1, 2).reshape(B, N, 3)
                gradient = gradient_base_frame
            return sdf, gradient

        if not hasattr(self, 'sdf_net'):
            self.get_model_dict(model)
            
        points_scaled = (points - self.offset) / self.scale  # Adjust points using offset and scale
        points_bounds = torch.where(points_scaled > 1.0 - 1e-2, 1.0 - 1e-2, points_scaled)
        points_bounds = torch.where(points_bounds < -1.0 + 1e-2, -1.0 + 1e-2, points_bounds)
        res_points = points_scaled - points_bounds

        if not use_derivative:
            with torch.no_grad():
                sdf_value = self.sdf_net(points_bounds)
            # sdf
            sdf = sdf_value + res_points.norm(dim=-1)
            sdf = sdf * self.scale
            return sdf, None
        else:
            points_bounds.requires_grad_(True)

            sdf_value = self.sdf_net(points_bounds)
            grad_sdf = torch.autograd.grad(sdf_value.sum(), points_bounds, create_graph=True)[0]

            sdf = sdf_value + res_points.norm(dim=-1)
            sdf = sdf * self.scale
            gradient = res_points + torch.nn.functional.normalize(grad_sdf, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()

            # Gradient transformation
            if base_trans is not None:
                # 从 base_trans 中提取旋转部分 (3x3)
                rotation_matrix = base_trans[:, :3, :3]  # (3, 3)

                # 将梯度从物体坐标系转换到基坐标系
                gradient_base_frame = torch.einsum('ijk,ikl->ijl', rotation_matrix, gradient.transpose(1, 2)).transpose(
                    1, 2).reshape(B, N, 3)
                gradient = gradient_base_frame
            return sdf, gradient

class SDF:
    def __init__(self, args, model_type="NN"):
        """
        Initialize the SDF class.
        """
        if model_type == "NN":
            self.model = SDFNN(args)
        elif model_type == "BP":
            self.model = SDFBP(args)
        self.wdf_dir = self.model.wdf_dir
        self.simple_shape = self.model.simple_shape
        self.device = self.model.device

    
    def train(self):
        self.model.train()
        
    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        self.model.create_surface_mesh(model, nbData, vis, save_mesh_name)
        
    def get_sdf_batch(self, points, model, base_trans=None, use_derivative=False):
        return self.model.get_sdf_batch(points, model, base_trans, use_derivative)
    
    