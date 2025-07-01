import os
import numpy as np
import torch
import trimesh
from typing import Union, List, Tuple


from robolab.coord import convert_ori_format, convert_quat_order, homo_matrix_from_quat_tensor
from robolab.formatter.mjcf_parser.mjcf import MJCF
from robolab.formatter.urdf_parser.urdf import URDF
from robolab.kinematics.fk import get_fk_from_chain
from robolab.kinematics.ik import get_ik_from_chain


class RobotModel:
    def __init__(self, model_path: str, robot_pose=None,
                 solve_engine: str = "pytorch_kinematics", device="cpu",
                 verbose=False, base_link="pelvis"):
        """
        Initialize a robot model from a URDF or MuJoCo XML file

        :param model_path: the path of the URDF or MuJoCo XML file
        :param robot_pose: initial pose of robot base link, [x, y, z, qx, qy, qz, qw]
        :param solve_engine: the engine to solve the kinematics, ["pytorch_kinematics", "kinpy", "all"]
        :param device: the device to run the computation
        :param verbose: whether to print the chain
        :param base_link: identify the name of the base link of the robot
        """
        assert solve_engine in ["pytorch_kinematics", "kinpy"], "Unsupported solve engine."
        self.solve_engine = solve_engine
        self.device = device
        self.verbose = verbose
        self.base_link = base_link
        self.robot_pose = robot_pose if robot_pose else [0, 0, 0, 0, 0, 0, 1]
        self.robot_pos = self.robot_pose[:3]
        self.robot_rot = self.robot_pose[3:]

        self.robot_model_path = model_path
        self.mesh_dir = os.path.join(os.path.dirname(model_path), "meshes")

        self._load_model()
        self._load_joint_info()
        self._load_mesh_info()
        self._load_link_info()

    def _load_model(self):
        """Loads the kinematic chain and robot model (URDF or MJCF)."""
        if self.solve_engine == "pytorch_kinematics":
            from robolab.kinematics import pytorch_kinematics_utils as pk_utils
            self.chain = pk_utils.build_chain_from_model(self.robot_model_path, self.verbose)
        elif self.solve_engine == "kinpy":
            from robolab.kinematics import kinpy_utils as kp_utils
            self.chain = kp_utils.build_chain_from_model(self.robot_model_path, self.verbose)

        if self.robot_model_path.endswith('.urdf'):
            self.robot_model = URDF.from_xml_file(self.robot_model_path)
        elif self.robot_model_path.endswith('.xml'):
            self.robot_model = MJCF(self.robot_model_path)
        else:
            raise ValueError("Unsupported model file format.")

    def _load_joint_info(self):
        """Loads joint information."""
        self.joint_list = self.get_joint_list()
        self.num_joint = len(self.joint_list)
        self.joint_limit_max = self.chain.high.to(self.device)
        self.joint_limit_min = self.chain.low.to(self.device)

    def _load_mesh_info(self):
        """Loads mesh information for the robot."""
        self.link_mesh_map = self.get_link_mesh_map()
        self.link_meshname_map = self.get_link_meshname_map()
        self.meshes, self.simple_shapes = self.load_meshes()
        self.robot_faces = [val[1] for val in self.meshes.values()]
        self.num_vertices_per_part = [val[0].shape[0] for val in self.meshes.values()]
        self.meshname_mesh = {key: val[0] for key, val in self.meshes.items()}
        self.meshname_mesh_normal = {key: val[-1] for key, val in self.meshes.items()}

    def _load_link_info(self):
        """Loads link information including virtual and real links."""
        self.link_virtual_map, self.inverse_link_virtual_map = self.get_link_virtual_map()
        self.real_link = self.get_real_link_list()
        self.all_link = self.get_link_list()

    def show_chain(self):
        beauty_print("Robot chain:")
        print(self.chain)

    def convert_to_serial_chain(self, export_link):
        import pytorch_kinematics as pk
        self.serial_chain = pk.SerialChain(self.chain, export_link)

    def set_init_pose(self, robot_pose):
        self.robot_pose = robot_pose
        self.robot_pos = robot_pose[:3]
        self.robot_rot = robot_pose[3:]

    def load_meshes(self):
        """
        Load all meshes and store them in a dictionary. Handles both complex meshes and simple shapes.

        :return: A dictionary where keys are mesh names and values are mesh data, and a dictionary for simple shapes.
        """
        meshes = {}
        simple_shapes = {}  # 用于保存简单形状信息
        for link_name, mesh_dict in self.link_mesh_map.items():
            for geom_name, mesh_info in mesh_dict.items():
                if mesh_info['type'] == 'mesh':
                    # 处理复杂的网格
                    mesh_file = mesh_info['params']['mesh_path']
                    name = mesh_info['params']['name']
                    mesh = trimesh.load(mesh_file)
                    if isinstance(mesh, trimesh.Scene):
                        combined_mesh = mesh.geometry.values()
                        mesh = trimesh.util.concatenate(combined_mesh)
                    temp = torch.ones(mesh.vertices.shape[0], 1).float().to(self.device)

                    vertices = torch.cat((torch.FloatTensor(np.array(mesh.vertices)).to(self.device), temp), dim=-1).to(
                        self.device)
                    normals = torch.cat((torch.FloatTensor(np.array(mesh.vertex_normals)).to(self.device), temp),
                                        dim=-1).to(self.device)

                    meshes[name] = [vertices, mesh.faces, normals]
                else:
                    # 处理简单几何形状，直接保存形状信息
                    simple_shapes[geom_name] = mesh_info

        return meshes, simple_shapes

    def get_joint_list(self):
        return self.chain.get_joint_parameter_names()

    def get_link_list(self):
        if self.solve_engine == "pytorch_kinematics":
            return self.chain.get_link_names()
        else:
            raise ValueError("kinpy does not support get_link_names() method.")

    def get_link_virtual_map(self):
        """
        :return: {link_body_name: [virtual_link_0, virtual_link_1, ...]}
        """
        all_links = self.get_link_list()
        link_virtual_map = {}
        for link in all_links:
            if "world" in link:
                continue
            if "_0" in link or "_1" in link or "_2" in link:
                link_name = link.split("_")[:-1]
                link_name = "_".join(link_name)
                if link_name not in link_virtual_map:
                    link_virtual_map[link_name] = []
                link_virtual_map[link_name].append(link)
            else:
                link_virtual_map[link] = [link]

        inverse_link_virtual_map = {v: k for k, vs in link_virtual_map.items() for v in vs}
        return link_virtual_map, inverse_link_virtual_map

    def get_real_link_list(self):
        """
        :return: [real_link_0, real_link_1, ...]
        """
        return list(self.link_virtual_map.keys())

    def get_link_mesh_map(self):
        """
        Get the map of link and its corresponding geometries from the robot model file (either URDF or MJCF).

        :return: {link_body_name: {geom_name: {'type': geom_type, 'params': geom_specific_parameters}}}

        If the robot model is a URDF file, it will attempt to link the geometry's mesh paths. The URDF format relies on
        external mesh files, and this function assumes that any `.obj` mesh files are converted to `.stl` files.

        If the robot model is an MJCF file (which has a `.xml` extension), it uses the MJCF-specific link-mesh mapping
        generated by the parser and processes different geometry types, including meshes, spheres, cylinders, boxes,
        and capsules.

        For each geometry type:
        - 'mesh': It maps the geometry name to its corresponding mesh file path.
        - 'sphere': It maps the geometry name to a dictionary with the sphere radius and position.
        - 'cylinder': It maps the geometry name to a dictionary with the cylinder's radius, height, and position.
        - 'box': It maps the geometry name to a dictionary with the box's extents (x, y, z) and position.
        - 'capsule': It maps the geometry name to a dictionary with the capsule's radius, height, and start/end positions.
        """
        if self.robot_model_path.endswith('.urdf'):
            link_mesh_map = {}
            for link, link_info in self.robot_model.link_map.items():
                visual = link_info.visual
                if visual is None or visual.geometry is None:
                    continue  # 跳过没有visual或geometry的link
                geometry = visual.geometry
                link_mesh_map[link] = {}
                if hasattr(geometry, "filename"):
                    mesh_path = geometry.filename
                    mesh_name = geometry.filename.split("/")[-1].split(".")[0]
                    link_mesh_map[link][mesh_name] = {
                        'type': 'mesh',
                        'params': {'mesh_path': os.path.join(os.path.dirname(self.robot_model_path), mesh_path),
                                'name': mesh_name, "scale": None}
                    }
                elif geometry.__class__.__name__.lower() == "sphere":
                    mesh_name = f"{link}_sphere"
                    link_mesh_map[link][mesh_name] = {
                        'type': 'sphere',
                        'params': {
                            'radius': geometry.radius,
                            'position': [0, 0, 0],
                            'name': mesh_name  
                        }
                    }
        elif self.robot_model_path.endswith('.xml'):
            link_mesh_map = self.robot_model.link_mesh_map
        else:
            raise ValueError("Unsupported model file.")
        return link_mesh_map

    def get_link_meshname_map(self):
        """
        :return: {link_body_name: [mesh_name]}
        """
        link_meshname_map = {}
        for link, geoms in self.link_mesh_map.items():
            link_meshname_map[link] = []
            for geom in geoms:
                link_meshname_map[link].append(self.link_mesh_map[link][geom]['params']['name'])
        return link_meshname_map

    def get_robot_mesh(self, vertices_list, faces):
        assert len(vertices_list) == len(faces), "The number of vertices and faces should be the same."
        robot_mesh = [trimesh.Trimesh(verts, face) for verts, face in zip(vertices_list, faces)]
        return robot_mesh

    # def get_forward_robot_mesh(self, joint_value, base_trans=None):
    #     """
    #     Transform the robot mesh according to the joint values and the base pose
    #     :param joint_value: the joint values, [batch_size, num_joint]
    #     :param base_trans: transformation matrix of the base pose, [batch_size, 4, 4]
    #     :return:
    #     """
    #     batch_size = joint_value.size()[0]
    #     outputs = self.forward(joint_value, base_trans)
    #     vertices_list = [[outputs[i][j].detach().cpu().numpy() for i in range(int(len(outputs) / 2))] for j in
    #                      range(batch_size)]
    #     mesh = [self.get_robot_mesh(vertices, self.robot_faces) for vertices in vertices_list]
    #     return mesh

    def get_forward_robot_mesh(self, joint_value, base_trans=None):
        """
        Transform the robot mesh according to the joint values and the base pose.
        Handles both complex meshes and simple shapes.

        :param joint_value: the joint values, [batch_size, num_joint]
        :param base_trans: transformation matrix of the base pose, [batch_size, 4, 4]
        :return: list of trimesh objects, one per batch
        """
        batch_size = joint_value.shape[0]
        outputs, transformed_shapes, trans_dict = self.forward(joint_value, base_trans)

        # 处理复杂网格和简单形状的 mesh
        mesh_list = []
        for j in range(batch_size):
            mesh_batch = []

            # 处理复杂网格
            for mesh_name, vertices in outputs.items():
                if mesh_name in self.meshes:
                    faces = self.meshes[mesh_name][1]  # 获取面索引
                    mesh = trimesh.Trimesh(vertices=vertices[j].detach().cpu().numpy(), faces=faces)
                    mesh_batch.append(mesh)

            # 处理简单形状
            for mesh_name, shape_info in transformed_shapes.items():
                # 获取关节的平移变换矩阵
                link_name = self.meshname_link_map[mesh_name]
                trans_matrix = trans_dict.get(link_name, None)
                if shape_info['type'] == 'sphere':
                    # 使用变换后的中心和半径创建球体
                    radius = shape_info['radius']
                    center = shape_info['center'][j].detach().cpu().numpy()
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
                    sphere.apply_translation(center)
                    # sphere.apply_transform(trans_matrix[j].detach().cpu().numpy())
                    mesh_batch.append(sphere)
                elif shape_info['type'] == 'box':
                    # 使用变换后的中心和边长创建立方体
                    extents = shape_info['extents']
                    center = shape_info['center'][j].detach().cpu().numpy()
                    box = trimesh.creation.box(extents=extents)
                    # box.apply_transform(trans_matrix[j].detach().cpu().numpy())
                    box.apply_translation(center)
                    mesh_batch.append(box)
                elif shape_info['type'] == 'cylinder':
                    # 使用变换后的中心、半径和高度创建圆柱体
                    radius = shape_info['radius']
                    height = shape_info['height']
                    center = shape_info['center'][j].detach().cpu().numpy()
                    cylinder = trimesh.creation.cylinder(radius=radius, height=height)
                    # cylinder.apply_transform(trans_matrix[j].detach().cpu().numpy())
                    cylinder.apply_translation(center)
                    mesh_batch.append(cylinder)
                elif shape_info['type'] == 'capsule':
                    # 使用变换后的中心、半径和高度创建胶囊体
                    radius = shape_info['radius']
                    height = shape_info['height']
                    center = shape_info['center'][j].detach().cpu().numpy()
                    capsule = trimesh.creation.capsule(radius=radius, height=height)
                    # capsule.apply_transform(trans_matrix[j].detach().cpu().numpy())
                    capsule.apply_translation(center)
                    mesh_batch.append(capsule)
            mesh_list.append(mesh_batch)

        return mesh_list

    def forward(self, joint_value, base_trans=None):
        """
        Transform the robot mesh according to the joint values and the base pose.
        Handles both complex meshes and simple shapes, including spheres, boxes, cylinders, and capsules.

        :param joint_value: the joint values, [batch_size, num_joint]
        :param base_trans: transformation matrix of the base pose, [batch_size, 4, 4]
        :return: transformed vertices and normals for complex meshes, and transformed parameters for simple shapes.
        """
        batch_size = joint_value.shape[0]
        trans_dict = self.get_trans_dict(joint_value, base_trans)
        self.meshname_link_map = {}
        for link, meshnames in self.link_meshname_map.items():
            for meshname in meshnames:
                self.meshname_link_map[meshname] = link

        ret_vertices = {}
        for mesh_name, mesh in self.meshname_mesh.items():
            link_vertices = self.meshname_mesh[mesh_name].repeat(batch_size, 1, 1)
            link_normals = self.meshname_mesh_normal[mesh_name].repeat(batch_size, 1, 1)

            if 'base' not in self.meshname_link_map[mesh_name]:
                link_name = self.meshname_link_map[mesh_name]
                related_link = [key for key in trans_dict.keys() if link_name in key][-1]
                link_vertices = torch.matmul(trans_dict[related_link], link_vertices.transpose(2, 1)).transpose(1, 2)[:,
                                :, :3]

            ret_vertices[mesh_name] = link_vertices

        # 存储简单形状的转换信息
        transformed_shapes = {}

        # 处理简单形状
        for mesh_name, shape_info in self.simple_shapes.items():
            link_name = self.meshname_link_map[mesh_name]
            if shape_info['type'] == 'sphere':
                radius = shape_info['params']['radius']
                center = torch.zeros(batch_size, 3).to(self.device)
                center = trans_dict[link_name][:, :3, 3].clone()
                center += torch.tensor(shape_info['params']['position'], dtype=torch.float32).to(self.device)
                center += torch.tensor([0, 0, shape_info['params']['radius']], dtype=torch.float32).to(self.device)
                transformed_shapes[mesh_name] = {'type': 'sphere', 'radius': radius, 'center': center}
            elif shape_info['type'] == 'box':
                extents = shape_info['params']['extents']
                center = torch.zeros(batch_size, 3).to(self.device)
                center = trans_dict[link_name][:, :3, 3].clone()
                center += torch.tensor(shape_info['params']['position'], dtype=torch.float32).to(self.device)
                transformed_shapes[mesh_name] = {'type': 'box', 'extents': extents, 'center': center}
            elif shape_info['type'] == 'cylinder':
                # 获取圆柱体的半径和高度
                radius = shape_info['params']['radius']
                height = shape_info['params']['height']
                center = torch.zeros(batch_size, 3).to(self.device)
                center = trans_dict[link_name][:, :3, 3].clone()
                center += torch.tensor(shape_info['params']['position'], dtype=torch.float32).to(self.device)
                transformed_shapes[mesh_name] = {'type': 'cylinder', 'radius': radius, 'height': height,
                                                 'center': center}

            elif shape_info['type'] == 'capsule':
                # 获取胶囊体的半径和高度
                radius = shape_info['params']['radius']
                height = shape_info['params']['height']
                center = torch.zeros(batch_size, 3).to(self.device)
                center = trans_dict[link_name][:, :3, 3].clone()
                center += torch.tensor(shape_info['params']['position'], dtype=torch.float32).to(self.device)
                transformed_shapes[mesh_name] = {'type': 'capsule', 'radius': radius, 'height': height,
                                                 'center': center}

        return ret_vertices, transformed_shapes, trans_dict

    # def forward(self, joint_value, base_trans=None):
    #     """
    #     Transform the robot mesh according to the joint values and the base pose
    #
    #     :param joint_value: the joint values, [batch_size, num_joint]
    #     :param base_trans: transformation matrix of the base pose, [batch_size, 4, 4]
    #     :return:
    #     """
    #     batch_size = joint_value.shape[0]
    #     trans_dict = self.get_trans_dict(joint_value, base_trans)
    #     meshname_link_map = {}
    #     for link, meshnames in self.link_meshname_map.items():
    #         for meshname in meshnames:
    #             meshname_link_map[meshname] = link
    #
    #     ret_vertices, ret_normals = [], []
    #     for mesh_name, mesh in self.meshname_mesh.items():
    #         link_vertices = self.meshname_mesh[mesh_name].repeat(batch_size, 1, 1)
    #         link_normals = self.meshname_mesh_normal[mesh_name].repeat(batch_size, 1, 1)
    #
    #         if 'base' not in meshname_link_map[mesh_name]:
    #             link_name = meshname_link_map[mesh_name]
    #             related_link = [key for key in trans_dict.keys() if link_name in key][-1]
    #             link_vertices = torch.matmul(trans_dict[related_link], link_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
    #             link_normals = torch.matmul(trans_dict[related_link], link_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
    #         else:
    #             if base_trans is not None:
    #                 link_vertices = torch.matmul(base_trans, link_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
    #                 link_normals = torch.matmul(base_trans, link_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
    #             else:
    #                 link_vertices = link_vertices[:, :, :3]
    #                 link_normals = link_normals[:, :, :3]
    #         ret_vertices.append(link_vertices)
    #         ret_normals.append(link_normals)
    #     return ret_vertices + ret_normals

    def get_fk(self, joint_value: Union[List, torch.Tensor], export_link=None):
        """
        Get the forward kinematics from a chain

        :param joint_value: both single and batch input are supported
        :param export_link: the name of the export link
        :return: the position, rotation of the end effector, and the transformation matrices of all links
        """
        joint_value = self._prepare_joint_value(joint_value)
        batch_size = joint_value.size()[0]

        if self.solve_engine == "pytorch_kinematics":
            return self._pytorch_fk(joint_value, export_link)
        elif self.solve_engine == "kinpy":
            return self._kinpy_fk(joint_value, export_link, batch_size)

    def _prepare_joint_value(self, joint_value: Union[List, torch.Tensor]):
        """Helper to prepare joint values for FK/IK."""
        if isinstance(joint_value, torch.Tensor):
            joint_value = joint_value.to(self.device)
        else:
            try:
                joint_value = torch.tensor(joint_value, dtype=torch.float32).to(self.device)
            except Exception as e:
                joint_value = torch.stack(joint_value, dim=0).to(self.device)
        if len(joint_value.size()) == 1:
            joint_value = joint_value.unsqueeze(0)
        return joint_value

    def _pytorch_fk(self, joint_value, export_link):
        """Helper function for PyTorch kinematics FK."""
        joint_value_dict = {joint: joint_value[:, i] for i, joint in enumerate(self.joint_list)}
        pose, ret = get_fk_from_chain(self.chain, joint_value_dict, export_link)

        if pose is not None:
            m = pose.get_matrix()
            pos = m[:, :3, 3]
            rot = convert_ori_format(m[:, :3, :3], "mat", "quat")
            return pos, rot, ret
        return None, None, ret

    def _kinpy_fk(self, joint_value, export_link, batch_size):
        """Helper function for KinPy kinematics FK."""
        pos_batch, rot_batch = [], []
        for batch in range(batch_size):
            joint_value_dict = {joint: joint_value[batch, i] for i, joint in enumerate(self.joint_list)}
            pose, ret = get_fk_from_chain(self.chain, joint_value_dict, export_link)
            if pose is not None:
                pos_batch.append(pose.pos)
                rot_batch.append(convert_quat_order(pose.rot, "wxyz", "xyzw"))
        return torch.tensor(pos_batch), torch.tensor(rot_batch), ret

    def get_jacobian(self, joint_value: List, export_link: str, locations=None):
        """
        Get the jacobian of a chain

        :param joint_value: the joint values, [batch_size, num_joint]
        :param export_link: the name of the export link
        :param locations: the locations offset from the export link
        :return:
        """
        self.convert_to_serial_chain(export_link=export_link)
        assert self.solve_engine == "pytorch_kinematics", "kinpy does not support get_jacobian() method."
        J = self.serial_chain.jacobian(joint_value, locations=locations)
        return J

    def get_trans_dict(self, joint_value: List, base_trans: Union[None, torch.Tensor] = None) -> dict:
        """
        Get the transformation matrices of all links

        :param joint_value: the joint values, [batch_size, num_joint]
        :param base_trans: transformation matrix of the base pose, [batch_size, 4, 4]
        :return: A dictionary where the keys are link names and the values are transformation matrices.
        """
        if base_trans is None:
            base_trans = (torch.from_numpy(np.identity(4)).to(self.device).
                          reshape(-1, 4, 4).expand(len(joint_value), 4, 4).float())

        _, _, ret = self.get_fk(joint_value)
        trans_dict = {}

        # Get the original base_link transformation matrix
        original_base_trans = ret[self.base_link].get_matrix().to(base_trans.device)
        original_base_trans_inv = torch.linalg.inv(original_base_trans)
        # Compute the relative transformation matrix that brings the original base_link to the new base_trans
        relative_trans = torch.matmul(base_trans, original_base_trans_inv)

        # Iterate over all links and apply the relative transformation if needed
        for link in self.all_link:
            if "world" in link:
                continue

            val = ret[link]
            homo_matrix = val.get_matrix().to(base_trans.device)

            real_link = self.inverse_link_virtual_map[link]

            # If base_trans is provided, apply the relative transformation to all links
            if base_trans is not None:
                # Update the link's transformation by applying the relative transformation
                homo_matrix = torch.matmul(relative_trans, homo_matrix)

            # Store the updated transformation in the dictionary
            trans_dict[real_link] = homo_matrix

        return trans_dict

    def get_ik(self, ee_pose: Union[torch.Tensor, None, List, Tuple], export_link, goal_in_rob_tf: bool = True,
               cur_configs=None, num_retries=10):
        """
        Get the inverse kinematics from a chain

        :param ee_pose: the pose of the end effector, 7D vector with the first 3 elements as position and the last 4 elements as rotation
        :param export_link: the name of the export link
        :param goal_in_rob_tf: whether the goal pose is in the robot base frame
        :param cur_configs: let the ik solver retry from these configurations
        :param num_retries: the number of retries
        :return: the joint values
        """
        self.convert_to_serial_chain(export_link)
        if self.solve_engine == "pytorch_kinematics":
            return get_ik_from_chain(self.serial_chain, ee_pose, self.device, goal_in_rob_tf=goal_in_rob_tf,
                                     robot_pose=self.robot_pose, cur_configs=cur_configs, num_retries=num_retries)
        elif self.solve_engine == "kinpy":
            import kinpy as kp
            self.serial_chain = kp.chain.SerialChain(self.chain, export_link)
            homo_matrix = homo_matrix_from_quat_tensor(ee_pose[3:], ee_pose[:3])
            return self.serial_chain.inverse_kinematics(homo_matrix)
