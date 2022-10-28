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

import os
import torch
import smplx

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from tasks.humanoid_ampmy import build_amp_observations, HumanoidAMPmy
import tasks.humanoid_amp_task as humanoid_amp_task
from tasks.mytask.utils_amp.motion_lib import MotionLib
from tasks.mytask.humanoid_amp_base import DOF_BODY_IDS, dof_to_obs, DOF_ACTIONS, compute_regular_reward

from isaacgym.torch_utils import *
from utils.torch_jit_utils import *

NUM_MOTION_TAR_PER_STEP = 13 + 63

class HumanoidCloning(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        super().__init__(   
                        cfg=cfg, 
                        rl_device=rl_device, 
                        sim_device=sim_device, 
                        graphics_device_id=graphics_device_id, 
                        headless=headless, 
                        virtual_screen_capture=virtual_screen_capture, 
                        force_render=force_render
                        )
                        
        pred_motion_file = cfg['task'].get('pred_motion_file', 'motion_run.yaml')
        pred_motion_root = cfg['task'].get('pred_motion_root', '/home/datassd/yuxuan/amass_with_babel_precomputed')
        pred_max_num = cfg['task'].get('pred_motion_num', None)
        pred_motion_file_path = os.path.join(pred_motion_root, pred_motion_file)
        # print(pred_motion_file_path)
        self._load_pred_motion(pred_motion_file_path, pred_max_num)
        
        self._termination_dist = self.cfg["env"]["terminationDist"]
        self.cloningprogress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        num_pred_motions = self._pred_motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._start_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._motion_ids = torch.remainder(self._motion_ids, num_pred_motions)#TODO
        self._motion_tar_buf = torch.zeros((self.num_envs, NUM_MOTION_TAR_PER_STEP), device=self.device, dtype=torch.float)
        body_model = cfg.get('body_model', 'smpl')
        body_model_path = cfg.get('body_model_path', '/home/datassd/yuxuan/smpl_model/models')
        self.body_model = smplx.create(model_path=body_model_path, model_type=body_model).to(self.device)
        return

    def _load_pred_motion(self, motion_file, max_num=None):
        key_body_ids_motionlib = np.array([23, 22, 11, 10, 15])
        
        self._pred_motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=key_body_ids_motionlib,
                                     max_num=max_num,
                                     device=self.device)
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 13 + 63
        return obs_size

    def _update_task_post(self):
        motion_ids = self._motion_ids.detach().cpu().numpy()
        motion_lengths = self._pred_motion_lib.get_motion_length(motion_ids)
        motion_lengths = torch.from_numpy(motion_lengths).to(self.device)
        progress = self.progress_buf.clone() - self.cloningprogress_buf
        reset_task_mask = compute_cloning_reset(
                                                self.reset_buf, 
                                                motion_lengths, 
                                                progress, 
                                                self._start_times, 
                                                self.dt,
                                                self._rigid_body_pos,
                                                self._enable_early_termination, 
                                                self._termination_height
                                                )
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._pred_motion_lib.get_motion_state(motion_ids, motion_times)

        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)

        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _compute_reset(self):
        tar_root_states = self._motion_tar_buf[:, :13]
        obs_root_states = self._humanoid_root_states
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   tar_root_states,
                                                   obs_root_states,
                                                   self.max_episode_length, self._enable_early_termination, self._termination_height,
                                                   self._termination_dist)
        return

    def _reset_actors(self, env_ids): #the function handle of reset, from public function reset_idx(self, env_ids) see above
        if (self._state_init == HumanoidAMPmy.StateInit.Start
              or self._state_init == HumanoidAMPmy.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.cloningprogress_buf[env_ids] = self.progress_buf[env_ids].clone()
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._pred_motion_lib.sample_motions(num_envs)
        
        if self._state_init == HumanoidAMPmy.StateInit.Random:
            motion_times = self._pred_motion_lib.sample_time(motion_ids) #random montionlib state
        elif (self._state_init == HumanoidAMPmy.StateInit.Start):
            motion_times = np.zeros(num_envs) #start motionlib state
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._pred_motion_lib.get_motion_state(motion_ids, motion_times)
        # if torch.isnan(root_pos).any():
        #     print('root_pos')
        #     raise
        # if torch.isnan(root_rot).any():
        #     print('root_rot')
        #     raise
        # if torch.isnan(root_vel).any():
        #     print('root_vel')
        #     raise
        # if torch.isnan(root_ang_vel).any():
        #     print('root_ang_vel')
        #     raise
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_ids[env_ids] = torch.from_numpy(motion_ids).to(self.device).type_as(self._motion_ids)
        self._start_times[env_ids] = torch.from_numpy(motion_times).to(self.device).type_as(self._start_times)
        return

    def _reset_task(self, env_ids):
        if (self._state_init == HumanoidAMPmy.StateInit.Start
              or self._state_init == HumanoidAMPmy.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        
        self._init_amp_obs(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        self.cloningprogress_buf[env_ids] = self.progress_buf[env_ids].clone()

        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        dt = self.dt
        if (env_ids is None):
            progress = self.progress_buf.clone() - self.cloningprogress_buf
            self._motion_tar_buf = self.obs_buf[:, -self.get_task_obs_size():]
            motion_ids = self._motion_ids
            motion_times = (progress + 1) * dt + self._start_times #important
        else:
            progress = self.progress_buf[env_ids].clone() - self.cloningprogress_buf[env_ids]
            self._motion_tar_buf[env_ids] = self.obs_buf[env_ids, -self.get_task_obs_size():]
            motion_ids = self._motion_ids[env_ids]
            motion_times = (progress + 1) * dt + self._start_times[env_ids]
        
        motion_ids = motion_ids.flatten().detach().cpu().numpy()
        motion_times = motion_times.flatten().detach().cpu().numpy()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._pred_motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        obs = compute_cloning_observations(root_states, dof_pos, dof_vel,
                                            key_pos, self._local_root_obs)
        
        return obs

    def _compute_reward(self, actions):
        tar_root_states = self._motion_tar_buf[:, :13]
        tar_dof_pos = self._motion_tar_buf[:, 13: 76]
        obs_root_states = self._humanoid_root_states
        obs_dof_pos = self._dof_pos
        body_model = self.body_model
        _dof_vel = self._dof_vel

        external_weight = 0.01
        self.rew_buf[:] = (1-external_weight) * compute_cloning_reward(tar_root_states, tar_dof_pos,
                                                 obs_root_states, obs_dof_pos,
                                                 body_model, _dof_vel)\
                            + external_weight * compute_regular_reward(actions[..., DOF_ACTIONS:])
        return

def compute_cloning_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    obs = torch.cat([root_states, dof_pos], dim=-1)
    return obs

def _mse_reward(a, b, scale):
    err = a - b
    se = torch.sum(err*err, dim=-1)
    reward = torch.exp(-scale * se)
    return reward

def compute_cloning_reward(tar_root_states, tar_dof_pos, root_states, dof_pos, body_model, dof_vel):#TODO beta

    key_joints_idx = [10, 11, 22, 23, 15]
    tar_rpos = tar_root_states[:, :3]
    tar_rrot = tar_root_states[:, 3:7]
    tar_rv = tar_root_states[:, 7:10]
    tar_rav = tar_root_states[:, 10:]
    rpos = root_states[:, :3]
    rrot = root_states[:, 3:7]
    rv = root_states[:, 7:10]
    rav = root_states[:, 10:]

    vel_err_scale = 0.25
    avel_err_scale = 0.25
    joints_err_scale = 5#1
    rot_err_scale = 5#1
    rpos_err_scale = 5#1
    regulate_err_scale = 0.01

    vel_reward_w = 0.1
    avel_reward_w = 0.3
    joints_reward_w = 1
    key_joints_reward_w = 3
    rot_reward_w = 1
    rpos_reward_w = 1
    regulate_reward_w = 0.05

    rv_reward = _mse_reward(tar_rv, rv, vel_err_scale)
    rav_reward = _mse_reward(tar_rav, rav, avel_err_scale)
    rpos_reward = _mse_reward(tar_rpos, rpos, rpos_err_scale)

    tar_rot = torch.cat([tar_rrot, tar_dof_pos], dim=-1)
    rot = torch.cat([rrot, dof_pos], dim=-1)
    # rot_reward = _mse_reward(tar_rot, rot, rot_err_scale)

    rot_smpl = torch.zeros(list(rot.shape[:-1]) + [24, 3], device=rot.device)
    rot_smpl[..., DOF_BODY_IDS, :] = rot[..., 4:].view(list(rot.shape[:-1]) + [21, 3])
    rot_smpl[..., 0, :] = quat_to_exp_map(rot[:, :4])
    # joints = body_model(global_orient=rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints
    joints = body_model(global_orient=tar_rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints #remove root rotation bias

    tar_rot_smpl = torch.zeros(list(tar_rot.shape[:-1]) + [24, 3], device=tar_rot.device)
    tar_rot_smpl[..., DOF_BODY_IDS, :] = tar_rot[..., 4:].view(list(tar_rot.shape[:-1]) + [21, 3])
    tar_rot_smpl[..., 0, :] = quat_to_exp_map(tar_rot[:, :4])
    tar_joints = body_model(global_orient=tar_rot_smpl[..., :1, :], body_pose=tar_rot_smpl[..., 1:, :], return_verts=False).joints
    
    joints_reward = _mse_reward(tar_joints.view([tar_joints.shape[0], -1]), joints.view([tar_joints.shape[0], -1]), joints_err_scale)

    key_joints_reward = _mse_reward(tar_joints[:, key_joints_idx].view([tar_joints.shape[0], -1]), joints[:, key_joints_idx].view([tar_joints.shape[0], -1]), joints_err_scale)


    tar_dof_obs = dof_to_obs(tar_dof_pos)
    tar_root_rot_obs = quat_to_tan_norm_yup(tar_rrot)
    tar_rot = torch.cat([tar_root_rot_obs, tar_dof_obs], dim=-1)

    dof_obs = dof_to_obs(dof_pos)
    root_rot_obs = quat_to_tan_norm_yup(rrot)
    rot = torch.cat([root_rot_obs, dof_obs], dim=-1)

    rot_reward = _mse_reward(tar_rot, rot, rot_err_scale)

    dof_vel_regular = _mse_reward(dof_vel, 0, regulate_err_scale)
    reward = vel_reward_w*rv_reward + avel_reward_w*rav_reward + rot_reward_w*rot_reward + rpos_reward_w*rpos_reward\
         + joints_reward_w*joints_reward + regulate_reward_w*dof_vel_regular + key_joints_reward_w*key_joints_reward#TODO weight
    reward /= vel_reward_w + avel_reward_w + rot_reward_w + rpos_reward_w + joints_reward_w + regulate_reward_w + key_joints_reward_w
    
    return reward
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_cloning_reset(reset_buf, motion_lengths, progress_buf, start_times, dt, \
    rigid_body_pos, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, bool, float) -> Tensor
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + start_times
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset

def compute_humanoid_reset(reset_buf, progress_buf, tar_root_states, root_states,
                           max_episode_length, enable_early_termination, termination_height, terminateion_dist):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, float, float) -> Tuple[Tensor, Tensor]
    tar_rpos = tar_root_states[:, :3]
    tar_rrot = tar_root_states[:, 3:7]
    tar_rv = tar_root_states[:, 7:10]
    tar_rav = tar_root_states[:, 10:]
    rpos = root_states[:, :3]
    rrot = root_states[:, 3:7]
    rv = root_states[:, 7:10]
    rav = root_states[:, 10:]

    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rpos[..., 1] #ROOT
        has_fallen = body_height < termination_height

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1) #1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        body_dist = torch.sum(torch.abs(tar_rpos[:, [0, 2]] - rpos[:, [0, 2]]), dim=-1)
        has_lost = body_dist > terminateion_dist
        has_lost *= (progress_buf > 1)
        terminated = torch.where(has_lost, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated