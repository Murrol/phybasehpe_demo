import os
import sys
import json
import joblib
import trimesh
import subprocess
import numpy as np
# from smplx import SMPL, SMPLH, SMPLX
import smplx
from smplx.joint_names import JOINT_NAMES
from matplotlib import cm as mpl_cm, colors as mpl_colors
from scipy.spatial import cKDTree
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion,\
    quaternion_to_axis_angle, quaternion_multiply, quaternion_raw_multiply, quaternion_to_matrix, \
    matrix_to_euler_angles, quaternion_invert, quaternion_apply


SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand'
    ]

SKELETON = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [9, 13],
    [9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 22],
    [21, 23]
]

PARENTS = [-1] + [x[0] for x in SKELETON]

JOINTS_CHILDLINK22 = {
    15: 'head',
    0: 'hips',
    16: 'leftArm',
    7: 'leftFoot',
    18: 'leftForeArm',
    20: 'leftHand',
    4: 'leftLeg',
    13: 'leftShoulder',
    10: 'leftToeBase',
    1: 'leftUpLeg',
    12: 'neck',
    17: 'rightArm',
    8: 'rightFoot',
    19: 'rightForeArm',
    21: 'rightHand',
    5: 'rightLeg',
    14: 'rightShoulder',
    11: 'rightToeBase',
    2: 'rightUpLeg',
    3: 'spine',
    6: 'spine1',
    9: 'spine2'
}

def quat_fk(lrot, lpos, parents): ###TODO
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quaternion_apply(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(quaternion_raw_multiply(gr[parents[i]], lrot[..., i:i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    return res


def quat_ik(grot, gpos, parents): ###TODO
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        torch.cat([
            grot[..., :1, :],
            quaternion_raw_multiply(quaternion_invert(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], dim=-2),
        torch.cat([
            gpos[..., :1, :],
            quaternion_apply(
                quaternion_invert(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], dim=-2)
    ]

    return res

def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):

        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors


def main(body_model='smpl', body_model_path='/home/datassd/yuxuan/smpl_model/models'):
    if body_model == 'smpl':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smpl/smpl_vert_segmentation.json')
    elif body_model == 'smplx':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smplx/smplx_vert_segmentation.json')
    elif body_model == 'smplh':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smpl/smpl_vert_segmentation.json')
    else:
        raise ValueError(f'{body_model} is not defined, \"smpl\", \"smplh\" or \"smplx\" are valid body models')

    body_model = smplx.create(model_path=body_model_path, model_type=body_model)
    part_segm = json.load(open(part_segm_filepath))
    
    part_segm['leftHand'] += part_segm.pop('leftHandIndex1')
    part_segm['rightHand'] += part_segm.pop('rightHandIndex1')#22 same as drecon
    print('num of parts:', len(part_segm.items()))




    vertices = body_model().vertices[0].detach().numpy()
    joints = body_model().joints[0].detach().numpy()[:24]
    loc_joints_quat = axis_angle_to_quaternion(torch.zeros(1, 24, 3))
    # print(loc_joints_quat)
    glob_joints_quat = loc_joints_quat
    glob_joints_pos = body_model().joints[:, :24]
    res = quat_ik(glob_joints_quat, glob_joints_pos, PARENTS)
    loc_joints_pos = res[1]

    recover = quat_fk(loc_joints_quat, loc_joints_pos, PARENTS)[1]
    # print(recover-glob_joints_pos) #verified

    # print(joints[0])

    # print(joints.shape)
    # print(len(JOINT_NAMES))
    # joints = [joints[JOINT_NAMES.index(n)] for n in SMPL_JOINT_NAMES]
    '''
    They should be: 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel' and the left, right hand finger tips. For SMPL the latter should not be very useful. You can actually see the order in the vertex_joint_selector script.
    '''
    print(joints.shape)
    faces = body_model.faces

    parts_mesh = list()
    save_dict = dict()
    bodyparts_CoM = dict()
    for part_idx, (k, v) in enumerate(part_segm.items()):
        mesh = trimesh.Trimesh(vertices[v], process=False)
        _mesh = mesh.convex_hull
        # parts_mesh.append(_mesh)
        _CoM = _mesh.center_mass #globol position
        bodyparts_CoM[k] = _CoM
        _mesh.vertices -= _CoM #align the CoM to origin
        save_dict[k] = _mesh
    # print(bodyparts_CoM)

    for k, v in JOINTS_CHILDLINK22.items(): #glob2loc
        # print(torch.from_numpy(bodyparts_CoM[v][None]) - glob_joints_pos[..., k, :])
        bodyparts_CoM[v] = quaternion_apply(
                quaternion_invert(glob_joints_quat[..., k, :]),
                torch.from_numpy(bodyparts_CoM[v][None]) - glob_joints_pos[..., k, :])[0].detach().numpy()
        # print(bodyparts_CoM[v])
    # print(bodyparts_CoM)
        

    for key, m in save_dict.items():
        m.export('../demo/body_parts_%s.obj' %key)

    # joblib.dump(save_dict, './body_parts.pkl')
    loc_joints_pos = loc_joints_pos.squeeze().detach().numpy()
    # print(loc_joints_pos)
    loc_joints_axisangle = np.zeros([24,3])
    # print(joints[[0,10,11,15]])
    print(loc_joints_pos[17])
    joblib.dump({'JOINTS_CHILDLINK22': JOINTS_CHILDLINK22, 'bodyparts_CoM': bodyparts_CoM, 'loc_angle': 0}, './links_info.pkl')
    joblib.dump({'skeleton': np.array(SKELETON), 'joints_name': SMPL_JOINT_NAMES, 'glob_joints_position': joints,\
         'loc_joints_pos': loc_joints_pos, 'loc_joints_axisangle': loc_joints_axisangle, 'parents':PARENTS}, \
        './joints_info.pkl')

    # parts = joblib.load('./body_parts.pkl')
    # os.makedirs('../demo', exist_ok=True)
    # for key, m in parts.items():
    #     m.export('../demo/body_parts_%s.stl' %key)
    # parts_mesh = joblib.load('./body_parts.pkl').values()
    # for idx, m in enumerate(parts_mesh):
    #     m.export('../demo/body_parts_%d.stl' %idx)
    # scene = trimesh.Scene(parts_mesh)
    # scene.show()
    # scene.export('../demo/body_parts.stl')
    # vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])
    # mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
    # mesh.show(background=(0,0,0,0))


if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2])
    main(body_model_path='/Users/yuxuanmu/project/smpl_model/models')