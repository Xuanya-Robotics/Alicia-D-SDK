import os

from robolab.formatter.mjcf_parser.parser import from_path
from robolab.utils import list_absl_path, create_dir


class MJCF:
    def __init__(self, file_path):
        self.link_mesh_map = {}

        self.robot = from_path(file_path)
        self.get_link_mesh_map()

    def get_link_mesh_map(self):
        """
        Get the map of link and its corresponding geometries from the MJCF file.

        :return: {link_body_name: {geom_name: mesh_path}}
        """
        bodies = self.robot.find_all("body")
        self.robot.compiler.meshdir = self.robot.compiler.meshdir or "meshes"
        mesh_dir = os.path.join(self.robot.namescope.model_dir, self.robot.compiler.meshdir)
        create_dir(mesh_dir)
        all_mesh_file_stl = list_absl_path(mesh_dir, recursive=True, suffix=".stl")
        all_mesh_file_STL = list_absl_path(mesh_dir, recursive=True, suffix=".STL")
        all_mesh_files = all_mesh_file_stl + all_mesh_file_STL

        mesh_map = self.robot.get_assets_map()
        mesh_name_path_map = {}
        for mesh_name, mesh_file in mesh_map.items():
            mesh_path = None
            for mesh_file_exist in all_mesh_files:
                if mesh_file in mesh_file_exist:
                    mesh_path = mesh_file_exist
            if mesh_path is not None:
                mesh_name_path_map[mesh_name] = mesh_path
            else:
                raise FileNotFoundError(f"Mesh file {mesh_file} not found in the mesh directory.")

        meshes = self.robot.find_all("mesh")

        # 遍历所有 bodies，处理几何体
        for body in bodies:
            geoms_this_body = body.geom
            self.link_mesh_map[body.name] = {}

            for geom in geoms_this_body:
                geom_type = geom.type or "capsule"  # 默认类型为胶囊
                geom_pos = geom.pos if geom.pos is not None else [0, 0, 0]

                # 处理不同的几何体类型
                if geom_type == "mesh":
                    geom_mesh_name = geom.mesh.name
                    geom_mesh_path = mesh_name_path_map[geom_mesh_name]
                    mesh_scale = [1, 1, 1]
                    for mesh in meshes:
                        if mesh.name == "wheelchair_mesh":
                            mesh_scale = mesh.scale
                    self.link_mesh_map[body.name][geom_mesh_name] = {
                        'type': 'mesh',
                        'params': {'mesh_path': geom_mesh_path, 'name': geom_mesh_name, 'position': geom_pos,
                                   'scale': mesh_scale}
                    }

                elif geom_type == "sphere":
                    geom_mesh_size = geom.size[0]  # 球体的大小是半径
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'sphere',
                        'params': {'radius': geom_mesh_size, 'position': geom_pos, 'name': geom.name}
                    }

                elif geom_type == "cylinder":
                    geom_mesh_size = geom.size  # 圆柱体的大小是 [半径, 高度]
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'cylinder',
                        'params': {'radius': geom_mesh_size[0], 'height': geom_mesh_size[1], 'position': geom_pos,
                                   'name': geom.name}
                    }

                elif geom_type == "box":
                    geom_mesh_size = geom.size  # 盒子的大小是 [x, y, z] 维度
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'box',
                        'params': {'extents': geom_mesh_size, 'position': geom_pos, 'name': geom.name}
                    }

                elif geom_type == "capsule":
                    geom_mesh_size = geom.size  # 胶囊的半径储存在 size[0]
                    geom_fromto = geom.fromto  # 从fromto属性获取胶囊两端的坐标
                    from_point = geom_fromto[:3]  # 胶囊起点
                    to_point = geom_fromto[3:]  # 胶囊终点
                    # 计算胶囊的高度（两点之间的距离）
                    height = ((to_point[0] - from_point[0]) ** 2 +

                              (to_point[1] - from_point[1]) ** 2 +

                              (to_point[2] - from_point[2]) ** 2) ** 0.5
                    # 胶囊的参数化描述
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'capsule',
                        'params': {
                            'radius': geom_mesh_size[0],  # 胶囊的半径
                            'height': height,  # 胶囊的高度
                            'from': from_point,  # 起点坐标
                            'to': to_point,  # 终点坐标
                            'name': geom.name,
                            "position": geom_pos
                        }
                    }

                else:
                    raise ValueError(f"Unsupported geometry type {geom_type}.")
        return self.link_mesh_map
