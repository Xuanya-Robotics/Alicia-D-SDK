import torch

from .transform_tensor import *


def convert_ori_format(ori, src_format: str, tar_format: str):
    """
    Convert orientation format from source to target format.

    :param ori: either quaternion, rotation matrix or euler angles
    :param src_format: source format
    :param tar_format: target format
    :return: converted orientation format
    """
    assert src_format in ['quat', 'mat', 'euler'], "Unsupported source format."
    assert tar_format in ['quat', 'mat', 'euler'], "Unsupported target format."
    if src_format == tar_format:
        return ori
    if src_format == 'quat':
        if tar_format == 'mat':
            return rot_matrix_from_quat_tensor(ori)
        elif tar_format == 'euler':
            return euler_from_quat_tensor(ori)
    elif src_format == 'mat':
        if tar_format == 'quat':
            return quat_from_rot_matrix_tensor(ori)
        elif tar_format == 'euler':
            return euler_from_rot_matrix_tensor(ori)
    elif src_format == 'euler':
        if tar_format == 'quat':
            return quat_from_euler_tensor(ori)
        elif tar_format == 'mat':
            return rot_matrix_from_euler_tensor(ori)


def convert_trans_format(trans, src_format, tar_format):
    """
    Convert transformation format from source to target format

    :param trans: transformation matrix or vector
    :param src_format: source format, ["trans_mat", "pos_quat", "pos_euler"]
    :param tar_format: target format, ["trans_mat", "pos_quat", "pos_euler"]
    :return: converted transformation
    """
    assert src_format in ['trans_mat', 'pos_quat', 'pos_euler'], "Unsupported source format."
    assert tar_format in ['trans_mat', 'pos_quat', 'pos_euler'], "Unsupported target format."

    if src_format == 'trans_mat':
        mat = trans[:, :3, :3]
        pos = trans[:, :3, 3]
        if tar_format == 'pos_quat':
            quat = quat_from_rot_matrix_tensor(mat)
            return torch.cat([pos, quat], dim=1)
        elif tar_format == 'pos_euler':
            euler = euler_from_rot_matrix_tensor(mat)
            return torch.cat([pos, euler], dim=1)
    elif src_format == 'pos_quat':
        pos = trans[:, :3]
        quat = trans[:, 3:]
        if tar_format == 'trans_mat':
            mat = rot_matrix_from_quat_tensor(quat)
            trans_matrix = torch.eye(4).repeat(trans.shape[0], 1, 1).to(trans.device)
            trans_matrix[:, :3, :3] = mat
            trans_matrix[:, :3, 3] = pos
            return trans_matrix
        elif tar_format == 'pos_euler':
            euler = euler_from_quat_tensor(quat)
            return torch.cat([pos, euler], dim=1)
    elif src_format == 'pos_euler':
        pos = trans[:, :3]
        euler = trans[:, 3:]
        if tar_format == 'trans_mat':
            mat = rot_matrix_from_euler_tensor(euler)
            trans_matrix = torch.eye(4).repeat(trans.shape[0], 1, 1)
            trans_matrix[:, :3, :3] = mat
            trans_matrix[:, :3, 3] = pos
            return trans_matrix
        elif tar_format == 'pos_quat':
            quat = quat_from_euler_tensor(euler)
            return torch.cat([pos, quat], dim=1)


def convert_quat_order(quat, src_order, tar_order):
    """
    Convert quaternion order from source to target order.
    Note: The quaternion order in robolab is 'xyzw'.

    :param quat: quaternion tensor
    :param src_order: source order, ['wxyz', 'xyzw']
    :param tar_order: target order, ['wxyz', 'xyzw']
    :return:
    """
    assert src_order in ['wxyz', 'xyzw'], "Unsupported source order."
    assert tar_order in ['wxyz', 'xyzw'], "Unsupported target order."
    quat = check_quat_tensor(quat)
    if src_order == tar_order:
        return quat
    if src_order == 'wxyz':
        if tar_order == 'xyzw':
            return quat[:, [1, 2, 3, 0]]
    elif src_order == 'xyzw':
        if tar_order == 'wxyz':
            return quat[:, [3, 0, 1, 2]]
