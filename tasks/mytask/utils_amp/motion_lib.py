# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from posixpath import dirname
from typing import Iterable
import numpy as np
import os
import yaml
import math
import time

from ..poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonMotion_dof
from ..poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from utils.torch_jit_utils import *

from tasks.mytask.humanoid_amp_base import DOF_BODY_IDS, DOF_OFFSETS


class MotionLib():
    def __init__(self, motion_file, num_dofs, key_body_ids, device, max_num=None):
        self._num_dof = num_dofs
        self._key_body_ids = key_body_ids
        self._device = device
        self._max_num = max_num
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motion_files), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motion_files)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        if isinstance(motion_id, Iterable):
            _motions = []
            for id in motion_id:
                _motions.append(SkeletonMotion_dof.from_file(self._motion_files[id]))
        else:
            _motions = SkeletonMotion_dof.from_file(self._motion_files[motion_id])
        return _motions

    def sample_motions(self, n):
        # motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len

        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()#actually num of joints
        num_key_bodies = self._key_body_ids.shape[0]

        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])
        root_rot1 = np.empty([n, 4])
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        local_rot0 = np.empty([n, num_bodies, 4])
        local_rot1 = np.empty([n, num_bodies, 4])
        dof_vel = np.empty([n, self._num_dof])
        key_pos0 = np.empty([n, num_key_bodies, 3])
        key_pos1 = np.empty([n, num_key_bodies, 3])

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        #process that will clip the motiontimes to [0,motion_len]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self._globalrootposs[f0l]
        root_pos1 = self._globalrootposs[f1l]

        root_rot0 = self._globalrootrots[f0l]
        root_rot1 = self._globalrootrots[f1l]

        local_rot0 = self._localrots[f0l]
        local_rot1 = self._localrots[f1l]

        root_vel = self._globalrootvs[f0l]
        root_ang_vel = self._globalrootavs[f0l]
        
        key_pos0 = self._globalkeyposs[f0l]
        key_pos1 = self._globalkeyposs[f1l]

        dof_vel = self._dofvs[f0l]

        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)

        root_pos0 = to_torch(root_pos0, device=self._device)
        root_pos1 = to_torch(root_pos1, device=self._device)
        root_rot0 = to_torch(root_rot0, device=self._device)
        root_rot1 = to_torch(root_rot1, device=self._device)
        root_vel = to_torch(root_vel, device=self._device)
        root_ang_vel = to_torch(root_ang_vel, device=self._device)
        local_rot0 = to_torch(local_rot0, device=self._device)
        local_rot1 = to_torch(local_rot1, device=self._device)
        key_pos0 = to_torch(key_pos0, device=self._device)
        key_pos1 = to_torch(key_pos1, device=self._device)
        dof_vel = to_torch(dof_vel, device=self._device)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))

        # start_time = time.time() #DEBUG TIME
        dof_pos = self._local_rotation_to_dof(local_rot)
        # print("_local_rotation_to_dof cost:", time.time()-start_time) #DEBUG TIME

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def _load_motions(self, motion_file):
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        _globalrootposs = []
        _globalrootrots = []
        _localrots = []
        _globalrootvs = []
        _globalrootavs = []
        _globalkeyposs = []
        _dofvs = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        
        if self._max_num:
            if self._max_num < num_motion_files:
                num_motion_files = self._max_num
        dir_name = os.path.dirname(motion_file)
        if 'precomputed' in dir_name:
            print('loading data from precomputed dataset')
            for f in range(num_motion_files):
                curr_file = motion_files[f]
                curr_weight = motion_weights[f]
                if curr_weight <= 0:
                    continue
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
                curr_motion = SkeletonMotion_dof.from_file(curr_file)
                motion_fps = curr_motion.fps
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                self._motion_lengths.append(curr_len)
                
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)

                _globalrootposs.append(curr_motion.global_translation[:, 0].numpy())
                
                _globalrootrots.append(curr_motion.global_rotation[:, 0].numpy())
            
                _localrots.append(curr_motion.local_rotation[:].numpy())
                
                _globalrootvs.append(curr_motion.global_root_velocity[:].numpy())
                _globalrootavs.append(curr_motion.global_root_angular_velocity[:].numpy())
                
                _globalkeyposs.append(curr_motion.global_translation[:, self._key_body_ids].numpy())

                _dofvs.append(curr_motion.dof_vels[:]) #precomputed in SkeletonMotion_dof and set the dof idx order correctly as motionlib
                
        else:
            assert(False), "Unsupported dataset folder, using 'precomputed' in dirname with local joint rotation angular velocity (dof_vel) precomputed."


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)
        
        num_bodies = self._get_num_bodies()#actually num of joints
        num_key_bodies = self._key_body_ids.shape[0]
        total_frames = np.sum(self._motion_num_frames)
        self._globalrootposs = np.empty([total_frames, 3])
        self._globalrootrots = np.empty([total_frames, 4])
        self._localrots = np.empty([total_frames, num_bodies, 4])
        self._globalrootvs = np.empty([total_frames, 3])
        self._globalrootavs = np.empty([total_frames, 3])
        self._globalkeyposs = np.empty([total_frames, num_key_bodies, 3])
        self._dofvs = np.empty([total_frames, self._num_dof])
 
        # self._globalrootposs = np.concatenate(_globalrootposs, axis=0)
        # self._globalrootrots = np.concatenate(_globalrootrots, axis=0)
        # self._localrots = np.concatenate(_localrots, axis=0)
        # self._globalrootvs = np.concatenate(_globalrootvs, axis=0)
        # self._globalrootavs = np.concatenate(_globalrootavs, axis=0)
        # self._globalkeyposs = np.concatenate(_globalkeyposs, axis=0)
        # self._dofvs = np.concatenate(_dofvs, axis=0)
        np.concatenate(_globalrootposs, out=self._globalrootposs, axis=0)
        np.concatenate(_globalrootrots, out=self._globalrootrots, axis=0)
        np.concatenate(_localrots, out=self._localrots, axis=0)
        np.concatenate(_globalrootvs, out=self._globalrootvs, axis=0)
        np.concatenate(_globalrootavs, out=self._globalrootavs, axis=0)
        np.concatenate(_globalkeyposs, out=self._globalkeyposs, axis=0)
        np.concatenate(_dofvs, out=self._dofvs, axis=0)

        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    ''' ##### old version get state in loop, load in list
    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()#actually num of joints
        num_key_bodies = self._key_body_ids.shape[0]

        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])
        root_rot1 = np.empty([n, 4])
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        local_rot0 = np.empty([n, num_bodies, 4])
        local_rot1 = np.empty([n, num_bodies, 4])
        dof_vel = np.empty([n, self._num_dof])
        key_pos0 = np.empty([n, num_key_bodies, 3])
        key_pos1 = np.empty([n, num_key_bodies, 3])

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :]  = curr_motion.global_translation[frame_idx0[ids], 0].numpy()
            root_pos1[ids, :]  = curr_motion.global_translation[frame_idx1[ids], 0].numpy()

            root_rot0[ids, :] = curr_motion.global_rotation[frame_idx0[ids], 0].numpy()
            root_rot1[ids, :]  = curr_motion.global_rotation[frame_idx1[ids], 0].numpy()

            local_rot0[ids, :, :]= curr_motion.local_rotation[frame_idx0[ids]].numpy()
            local_rot1[ids, :, :] = curr_motion.local_rotation[frame_idx1[ids]].numpy()

            root_vel[ids, :] = curr_motion.global_root_velocity[frame_idx0[ids]].numpy()
            root_ang_vel[ids, :] = curr_motion.global_root_angular_velocity[frame_idx0[ids]].numpy()
            
            key_pos0[ids, :, :] = curr_motion.global_translation[frame_idx0[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]].numpy()
            key_pos1[ids, :, :] = curr_motion.global_translation[frame_idx1[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]].numpy()

            dof_vel[ids, :] = curr_motion.dof_vels[frame_idx0[ids]]

        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)

        root_pos0 = to_torch(root_pos0, device=self._device)
        root_pos1 = to_torch(root_pos1, device=self._device)
        root_rot0 = to_torch(root_rot0, device=self._device)
        root_rot1 = to_torch(root_rot1, device=self._device)
        root_vel = to_torch(root_vel, device=self._device)
        root_ang_vel = to_torch(root_ang_vel, device=self._device)
        local_rot0 = to_torch(local_rot0, device=self._device)
        local_rot1 = to_torch(local_rot1, device=self._device)
        key_pos0 = to_torch(key_pos0, device=self._device)
        key_pos1 = to_torch(key_pos1, device=self._device)
        dof_vel = to_torch(dof_vel, device=self._device)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        
        if self._max_num:
            if self._max_num < num_motion_files:
                num_motion_files = self._max_num
        dir_name = os.path.dirname(motion_file)
        if 'precomputed' in dir_name:
            print('loading data from precomputed dataset')
            for f in range(num_motion_files):
                curr_file = motion_files[f]
                curr_weight = motion_weights[f]
                if curr_weight <= 0:
                    continue
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
                curr_motion = SkeletonMotion_dof.from_file(curr_file)
                motion_fps = curr_motion.fps
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                self._motions.append(curr_motion)
                self._motion_lengths.append(curr_len)
                
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)
        else:
            print('loading data from vanilla dataset')
            for f in range(num_motion_files):
                curr_file = motion_files[f]
                curr_weight = motion_weights[f]
                if curr_weight <= 0:
                    continue
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
                curr_motion = SkeletonMotion.from_file(curr_file)
                motion_fps = curr_motion.fps
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)
    
                curr_dof_vels = self._compute_motion_dof_vels(curr_motion) #time consuming
                curr_motion.dof_vels = curr_dof_vels

                self._motions.append(curr_motion)
                self._motion_lengths.append(curr_len)
                
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return
    '''
    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).astype(np.int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = np.array(dof_vels)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = quat_to_exp_map(joint_q) #NOTE isaac gym automatically use exp map for spherical joint rather than euler angle 
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
                # joint_euler = torch.stack(get_euler_xyz(joint_q), dim=-1)
                # joint_euler = torch.where(joint_euler>math.pi, joint_euler-2*math.pi, joint_euler)
                # print(joint_euler)
                # dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_euler
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                #TODO better realign the theta to specific axis(try to use the max axis), but now it should still work
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis #NOTE just for knee in zup system

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        dof_vel = np.zeros([self._num_dof])

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        # joint_euler = torch.stack(get_euler_xyz(diff_quat_data), dim=-1)
        # joint_euler = torch.where(joint_euler>math.pi, joint_euler-2*math.pi, joint_euler)
        # local_vel = joint_euler/ dt
        local_vel = local_vel.numpy()

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                # print(local_vel[body_id])
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis #TODO

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel