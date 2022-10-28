import numpy as np
import joblib

import glob
from tqdm import tqdm
import os
from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import *
from poselib.core.rotation3d import *
import torch
import math


def exp_map_to_angle_axis_yup(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., 1] = 1

    mask = angle > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

def main():
    # load motion raw
    skeleton_info = joblib.load('joints_info.pkl')

    root_path = '/home/datassd/yuxuan'
    amass_path = 'amass_data_unified_999_yup'
    subdataset = ''#'CMU'
    file_list = [x for x in glob.glob(os.path.join(root_path, amass_path) + '/**/**/*_poses.npz') if subdataset in x]

    angles = np.zeros((2,24,3))
    for f_path in tqdm(file_list):
        p = list(os.path.split(f_path))

        p[0] = p[0].replace(amass_path, amass_path+'_motionlib')
        f_d = np.load(f_path)
        local_rotation = f_d['poses'].reshape(-1, 3) #rotvec, namely angle axis
        q = exp_map_to_angle_axis_yup(torch.from_numpy(local_rotation))
        q = quat_from_angle_axis(*q)

        r = R.from_rotvec(local_rotation)
        print(q[0], r.as_quat()[0])
        print(torch.stack(get_euler_xyz(q), dim=-1)[0], r.as_euler('xyz')[0])
        euler_tst = torch.stack(get_euler_xyz(q), dim=-1)
        euler_tst = torch.where(euler_tst>math.pi, euler_tst-2*math.pi, euler_tst)
        print(torch.sum(torch.abs(euler_tst-torch.from_numpy(r.as_euler('xyz')))))
        # print(torch.stack(get_euler_xyz(q), dim=-1).min(dim=0))#0-2pi
        

        euler = r.as_euler('xyz').reshape(-1, 24, 3)
        euler_min = np.min(euler, axis=0)
        euler_max = np.max(euler, axis=0)
        
        angles[0] = np.where(angles[0]< euler_min, angles[0], euler_min)
        angles[1] = np.where(angles[1]> euler_max, angles[1], euler_max)
        break
    # print(angles)
    # np.save('joint_limits.npy', angles)
    return

if __name__ == '__main__':
    main()