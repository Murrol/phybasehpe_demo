from typing import Iterable
import torch 
import numpy as np
import smplx
import glob
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    # hands_pose = np.zeros([1,6]) #set hand pose to zeros 
    raw_amass_path = '/Users/yuxuanmu/project/amass_data'
    body_model_path='/Users/yuxuanmu/project/smpl_model/models'
    fps = 999#20.
    count = 0
    
    zup2yup = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    r1 = R.align_vectors(np.eye(3), zup2yup)[0]
    r1 = r1.as_matrix()
    file_list = glob.glob(raw_amass_path + '/**/**/*_poses.npz')
    for f_path in tqdm(file_list):
        p = list(os.path.split(f_path))
        p[0] = p[0].replace('amass_data', 'amass_data_unified_%d_yup' %fps)
        f_d = np.load(f_path)
        current_fps = f_d['mocap_framerate'] #60,100,120,250
        if fps == 999:
            sample_rate = 1
        else:
            if np.round(current_fps) % fps > 0.01:
                print(current_fps)
                continue
            sample_rate = int(current_fps // fps)
        trans = f_d['trans'][::sample_rate][:-1] # z upper 
        # print(np.var(trans[:,2]))
        poses = f_d['poses'][::sample_rate][:-1]
        betas = f_d['betas'][:10]
        smplh2smpl = list(range(0, 23*3)) + list(range(36*3, 37*3))
        poses = poses[:, smplh2smpl]
        N = poses.shape[0]
        if N<20:
            continue
        body_model = smplx.create(model_path=body_model_path, model_type='smpl', betas=betas[None], global_orient=poses[:1, :3], body_pose=poses[:1, 3:])
        joints = body_model().joints.detach().numpy()[0, :24]

        H_cam = np.eye(4)
        H_cam[0:3, 0:3] = r1.copy()  # [4, 4]
        H_cam = H_cam[None].repeat(N, axis=0)  # [T, 4, 4]

        r_root = R.from_rotvec(poses[:, :3]).as_matrix()  # [T, 3, 3]
        t_root = joints[:1]  # [1, 3]
        H_root = np.eye(4)[None].repeat(N, axis=0)  # [T, 4, 4]
        H_root[:, 0:3, 0:3] = r_root
        H_root[:, 0:3, 3] = trans + t_root

        H = np.einsum('Bij,Bjk ->Bik', H_cam, H_root)
        new_global_orient = R.from_matrix(H[:, 0:3, 0:3]).as_rotvec()  # [T, 3]
        new_trans = H[:, 0:3, 3] - t_root
        
        trans = new_trans
        # print(np.mean(trans, axis=0))
        poses[:, :3] = new_global_orient
        # print(joints[:, 0])
        # break
        os.makedirs(p[0], exist_ok=True)
        # body_model = smplx.create(model_path=body_model_path, model_type='smpl', betas=betas[None], global_orient=poses[:1, :3], body_pose=poses[:1, 3:])
        # joints = body_model().joints.detach().numpy()[0, np.array([10,11,15])] + trans[0]
        # print(joints)
        if fps == 999:
            np.savez(os.path.join(p[0], p[1]), trans=trans, poses=poses, gender=f_d['gender'], betas=betas, fps=current_fps)
        else:
            np.savez(os.path.join(p[0], p[1]), trans=trans, poses=poses, gender=f_d['gender'], betas=betas, fps=fps)
        count += 1
    print(count)
        

        


    