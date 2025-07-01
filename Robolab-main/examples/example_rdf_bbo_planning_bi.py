"""
Bimanual box carrying using Robot distance field (RDF)
========================

This example plans the contacts of a bimanual box carrying task via optimization based on RDF.
"""
import argparse

import numpy as np
import torch
import trimesh

import rofunc as rf


def box_carrying_contact_rdf(args):
    box_size = np.array([0.18, 0.1, 0.16])
    # box_size = np.array([0.28, 0.2, 0.26])
    box_pos = np.array([0.2, 0.0, 0.7])
    # box_pos = np.array([0.7, 0.2, 1.3])
    box_rotation = rf.robolab.convert_ori_format([0, 1, 1.57], "euler", "mat").numpy()[0]
    # box_rotation = np.array([[0, 1.57, 0],
    #                          [-1.57, 0, 0],
    #                          [0, 0, 1]])

    rdf_model = torch.load(args.rdf_model_path)

    # used_links = ["shell", "head_link1", "head_link2", "panda_left_link1", "panda_left_link2", "panda_left_link3",
    #               "panda_left_link4", "panda_left_link5", "panda_left_link6", "panda_left_link7", "panda_right_link1",
    #               "panda_right_link2", "panda_right_link3", "panda_right_link4", "panda_right_link5",
    #               "panda_right_link6", "panda_right_link7"]
    used_links = ["shoulder_pitch_link_r", "shoulder_roll_link_r", "elbow_pitch_link_r",
                  "shoulder_pitch_link_l", "shoulder_roll_link_l", "elbow_pitch_link_l"]
    bbo_planner = rf.robolab.rdf.BBOPlannerBi(args, rdf_model, box_size, box_pos, box_rotation, used_links=used_links)
    num_joint = bbo_planner.rdf_bp.robot.num_joint

    # contact points
    contact_points = bbo_planner.contact_points
    p_l, p_r, n_l, n_r = contact_points[0], contact_points[1], contact_points[2], contact_points[3]

    # initial joint value
    joint_max = bbo_planner.rdf_bp.robot.joint_limit_max
    joint_min = bbo_planner.rdf_bp.robot.joint_limit_min
    # mid = torch.rand(num_joint).to(args.device) * (joint_max - joint_min) + joint_min
    mid = torch.ones(num_joint).to(args.device) * 0.5 * (joint_max + joint_min)
    # mid = torch.zeros(num_joint).to(args.device)

    # visualize planning results
    zeros = torch.zeros(num_joint).to(args.device)
    base_pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).float()

    scene = trimesh.Scene()
    pc1 = trimesh.PointCloud(bbo_planner.object_internal_points.detach().cpu().numpy(), colors=[0, 255, 0])
    pc2 = trimesh.PointCloud(p_l.detach().cpu().numpy(), colors=[255, 0, 0])
    pc3 = trimesh.PointCloud(p_r.detach().cpu().numpy(), colors=[255, 0, 0])
    scene.add_geometry([pc1, pc2, pc3])
    scene.add_geometry(bbo_planner.object_mesh)
    robot = bbo_planner.rdf_bp.robot.get_forward_robot_mesh(mid.reshape(-1, num_joint), base_pose)[0]
    robot = np.sum(robot)
    robot.visual.face_colors = [150, 150, 200, 200]
    scene.add_geometry(robot, node_name='robot')
    scene.show()

    base_pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).float()
    joint_value = bbo_planner.bi_optimizer(p_l, p_r, n_l, n_r, mid, base_trans=base_pose, batch=128)
    joint_conf = {
        'joint_value': joint_value,
    }
    torch.save(joint_conf, args.joint_conf_path)

    # load planned joint conf
    data = torch.load(args.joint_conf_path)
    joint_value = data['joint_value']
    print('joint_value', joint_value.shape)

    # visualize planning results
    scene = trimesh.Scene()
    pc1 = trimesh.PointCloud(bbo_planner.object_internal_points.detach().cpu().numpy(), colors=[0, 255, 0])
    pc2 = trimesh.PointCloud(p_l.detach().cpu().numpy(), colors=[255, 0, 0])
    pc3 = trimesh.PointCloud(p_r.detach().cpu().numpy(), colors=[255, 0, 0])
    scene.add_geometry([pc1, pc2, pc3])
    scene.add_geometry(bbo_planner.object_mesh)

    # visualize the final joint configuration
    for t in joint_value:
        print('Joint value:', t)
        # ['hip_yaw_r', 'hip_roll_r', 'hip_pitch_r', 'knee_pitch_r', 'ankle_pitch_r', 'hip_yaw_l', 'hip_roll_l',
        #  'hip_pitch_l', 'knee_pitch_l', 'ankle_pitch_l', 'shoulder_pitch_r', 'shoulder_roll_r', 'elbow_pitch_r',
        #  'shoulder_pitch_l', 'shoulder_roll_l', 'elbow_pitch_l']
        t[:10] = 0

        robot = bbo_planner.rdf_bp.robot.get_forward_robot_mesh(t.reshape(-1, num_joint), base_pose)[0]
        robot = np.sum(robot)
        robot.visual.face_colors = [150, 150, 200, 200]
        scene.add_geometry(robot, node_name='robot')
        scene.show()

        scene.delete_geometry('robot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--num_results', default=3, type=int)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train_epoch', default=200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save_mesh_dict', action='store_false')
    parser.add_argument('--parallel', action='store_true')
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/mjcf/alicia", type=str)
    # parser.add_argument('--robot_asset_name', default="Alicia_0624.xml", type=str)
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/urdf/franka_description", type=str)
    # parser.add_argument('--robot_asset_name', default="robots/franka_panda.urdf", type=str)
    parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/mjcf/bruce", type=str)
    parser.add_argument('--robot_asset_name', default="bruce.xml", type=str)
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/mjcf/curi", type=str)
    # parser.add_argument('--robot_asset_name', default="curi_isaacgym_rdf.xml", type=str)
    parser.add_argument('--rdf_model_path', default=None)
    parser.add_argument('--joint_conf_path', default=None)
    parser.add_argument('--sampled_points_dir', default=None, type=str)
    args = parser.parse_args()
    args.rdf_model_path = f"{args.robot_asset_root}/rdf/BP/BP_{args.n_func}.pt"
    args.joint_conf_path = f"{args.robot_asset_root}/rdf/BP/joint_conf.pt"

    box_carrying_contact_rdf(args)
