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

class HumanoidCloning_pretrain(humanoid_amp_task.HumanoidAMPTask):
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

        self._termination_dist = self.cfg["env"]["terminationDist"]
        self.cloningprogress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        num_pred_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._start_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._motion_ids = torch.remainder(self._motion_ids, num_pred_motions)#TODO
        self._motion_tar_buf = torch.zeros((self.num_envs, 2, NUM_MOTION_TAR_PER_STEP), device=self.device, dtype=torch.float)
        self._motion_tar_curr = self._motion_tar_buf[:, 0]
        self._motion_tar_next = self._motion_tar_buf[:, 1]
        body_model = cfg.get('body_model', 'smpl')
        body_model_path = cfg.get('body_model_path', '/home/datassd/yuxuan/smpl_model/models')
        self.body_model = smplx.create(model_path=body_model_path, model_type=body_model).to(self.device)
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            # obs_size = 13 + 63
            obs_size = 808 #for max cloning obs
        return obs_size

    def post_physics_step(self):
        super().post_physics_step()
        return

    def _update_task_post(self):
        motion_ids = self._motion_ids.detach().cpu().numpy()
        motion_lengths = self._motion_lib.get_motion_length(motion_ids)
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
        # dt = self.dt
        # motion_ids = self._motion_ids
        # motion_times = self.progress_buf * dt + self._start_times
        # motion_ids = motion_ids.flatten()
        # motion_times = motion_times.flatten()
        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
        #        = self._pred_motion_lib.get_motion_state(motion_ids, motion_times)
        # root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        # self._motion_tar_buf = torch.cat([root_states, dof_pos, dof_vel, key_pos])
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

    def _compute_reset(self):
        tar_root_states = self._motion_tar_curr[:, :13]
        obs_root_states = self._humanoid_root_states
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   tar_root_states,
                                                   obs_root_states,
                                                   self.max_episode_length, self._enable_early_termination, self._termination_height,
                                                   self._termination_dist)
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)

        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
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
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if self._state_init == HumanoidAMPmy.StateInit.Random:
            motion_times = self._motion_lib.sample_time(motion_ids) #random montionlib state
        elif (self._state_init == HumanoidAMPmy.StateInit.Start):
            motion_times = np.zeros(num_envs) #start motionlib state
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
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
        
        if torch.isnan(self.obs_buf).any():
            self.obs_buf[:] = torch.where(torch.isnan(self.obs_buf), torch.zeros_like(self.obs_buf), self.obs_buf)
        return

    def _compute_task_obs(self, env_ids=None):
        dt = self.dt
        if (env_ids is None):
            cur_root_states = self._humanoid_root_states
            cur_dof_pos = self._dof_pos
            cur_dof_vel = self._dof_vel
            progress = self.progress_buf.clone() - self.cloningprogress_buf
            self._motion_tar_curr = self._motion_tar_next
            motion_ids = self._motion_ids
            motion_times = (progress + 1) * dt + self._start_times #important
        else:
            cur_root_states = self._humanoid_root_states[env_ids]
            cur_dof_pos = self._dof_pos[env_ids]
            cur_dof_vel = self._dof_vel[env_ids]
            progress = self.progress_buf[env_ids].clone() - self.cloningprogress_buf[env_ids]
            self._motion_tar_curr[env_ids] = self._motion_tar_next[env_ids]
            
            motion_ids = self._motion_ids[env_ids]
            motion_times = (progress + 1) * dt + self._start_times[env_ids]
        
        motion_ids = motion_ids.flatten().detach().cpu().numpy()
        motion_times = motion_times.flatten().detach().cpu().numpy()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        if (env_ids is None):
            self._motion_tar_next[:] = torch.cat([root_states, dof_pos], dim=-1)
        else:
            self._motion_tar_next[env_ids] = torch.cat([root_states, dof_pos], dim=-1)
        # obs = compute_cloning_observations(root_states, dof_pos, dof_vel,
        #                                     key_pos, self._local_root_obs)
        obs = compute_cloning_observations_max(root_states, dof_pos, dof_vel,
                                              cur_root_states, cur_dof_pos, cur_dof_vel,
                                              self.body_model,
                                              self._local_root_obs) #TODO
        
        return obs

    def _compute_reward(self, actions):

        tar_root_states = self._motion_tar_curr[:, :13]
        tar_dof_pos = self._motion_tar_curr[:, 13: 76]
        obs_root_states = self._humanoid_root_states
        obs_dof_pos = self._dof_pos
        body_model = self.body_model
        _dof_vel = self._dof_vel

        external_weight = 0.01
        self.rew_buf[:] = (1-external_weight) * compute_cloning_reward(tar_root_states, tar_dof_pos,
                                                 obs_root_states, obs_dof_pos,
                                                 body_model, _dof_vel)\
                            + external_weight * compute_regular_reward(actions[..., DOF_ACTIONS:])
        if torch.isnan(self.rew_buf).any():
            self.rew_buf[:] = torch.where(torch.isnan(self.rew_buf), torch.zeros_like(self.rew_buf), self.rew_buf)
        return

def compute_cloning_observations_max(tar_root_states, tar_dof_pos, tar_dof_vel, root_states, dof_pos, dof_vel, body_model, local_root_obs):
    key_joints_idx = [10, 11, 22, 23, 15]
    obs = []
    tar_rpos = tar_root_states[:, :3]
    tar_rrot = tar_root_states[:, 3:7]
    tar_rv = tar_root_states[:, 7:10]
    tar_rav = tar_root_states[:, 10:]
    rpos = root_states[:, :3]
    rrot = root_states[:, 3:7]
    rv = root_states[:, 7:10]
    rav = root_states[:, 10:]

    #cur heading quat (1, 4)
    hq = calc_heading_quat_yup(rrot)
    tar_hq = calc_heading_quat_yup(tar_rrot)
    obs += [hq]

    #heading rot angle (1, 3)
    hrot = calc_heading_yup(rrot)[:, None]
    tar_hrot = calc_heading_yup(tar_rrot)[:, None]
    diff_hrot = tar_hrot - hrot
    obs += [tar_hrot, hrot, diff_hrot]

    #root height (1, 3)
    tar_h = tar_rpos[:, 1:2]
    cur_h = rpos[:, 1:2] #duplicated in humanoid obs max
    diff_h = tar_h - cur_h
    obs += [tar_h, cur_h, diff_h]

    #dof and joint rot relative (1, 21*3*2 + 6 = 132 * 3)
    tar_dof_obs = dof_to_obs(tar_dof_pos)
    tar_root_rot_obs = quat_to_tan_norm_yup(tar_rrot)
    tar_rot_obs = torch.cat([tar_root_rot_obs, tar_dof_obs], dim=-1)

    dof_obs = dof_to_obs(dof_pos)
    root_rot_obs = quat_to_tan_norm_yup(rrot)
    rot_obs = torch.cat([root_rot_obs, dof_obs], dim=-1)

    rot_diff_obs = tar_rot_obs - rot_obs

    obs += [tar_rot_obs, rot_obs, rot_diff_obs]

    #dof vel diff (1, 63)
    dof_vel_diff = tar_dof_vel - dof_vel
    obs += [dof_vel_diff.flatten(start_dim=1)]

    #global root vel and avel (1, 18)
    rv_diff = tar_rv - rv
    rav_diff = tar_rav - rav
    obs += [tar_rv, rv, rv_diff, tar_rav, rav, rav_diff]

    #local root vel and avel (1, 6)
    local_root_vel = quat_rotate_inverse(hq, rv)
    local_root_ang_vel = quat_rotate_inverse(hq, rav)

    tar_local_root_vel = quat_rotate_inverse(tar_hq, tar_rv)
    tar_local_root_ang_vel = quat_rotate_inverse(tar_hq, tar_rav)

    local_root_vel_diff = tar_local_root_vel - local_root_vel
    local_root_ang_vel_diff = tar_local_root_ang_vel - local_root_ang_vel

    obs += [local_root_vel_diff, local_root_ang_vel_diff]

    #relative joint (1, 45*3*2 + 5*3*3 = 315)
    tar_rot = torch.cat([tar_rrot, tar_dof_pos], dim=-1)
    rot = torch.cat([rrot, dof_pos], dim=-1)

    rot_smpl = torch.zeros(list(rot.shape[:-1]) + [24, 3], device=rot.device)
    rot_smpl[..., DOF_BODY_IDS, :] = rot[..., 4:].view(list(rot.shape[:-1]) + [21, 3])
    rot_smpl[..., 0, :] = quat_to_exp_map(rot[:, :4])
    joints = body_model(global_orient=rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints #remove root rotation bias

    tar_rot_smpl = torch.zeros(list(tar_rot.shape[:-1]) + [24, 3], device=tar_rot.device)
    tar_rot_smpl[..., DOF_BODY_IDS, :] = tar_rot[..., 4:].view(list(tar_rot.shape[:-1]) + [21, 3])
    tar_rot_smpl[..., 0, :] = quat_to_exp_map(tar_rot[:, :4])
    tar_joints = body_model(global_orient=rot_smpl[..., :1, :], body_pose=tar_rot_smpl[..., 1:, :], return_verts=False).joints
    
    joints_diff = tar_joints.flatten(start_dim=1) - joints.flatten(start_dim=1)
    key_joints_diff = tar_joints[:, key_joints_idx].flatten(start_dim=1) - joints[:, key_joints_idx].flatten(start_dim=1)
    
    obs += [joints.flatten(start_dim=1), joints_diff, 
            tar_joints[:, key_joints_idx].flatten(start_dim=1), joints[:, key_joints_idx].flatten(start_dim=1), key_joints_diff]

    obs = torch.cat(obs, dim=-1)
    # print(obs.shape)
    return obs

def compute_cloning_reward(tar_root_states, tar_dof_pos, root_states, dof_pos, body_model, dof_vel):#beta completed $TODO scale of the erro
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
    avel_err_scale = 0.05
    rpos_err_scale = 10
    joints_err_scale = 1
    kjoints_err_scale = 10
    rot_err_scale = 5
    regulate_err_scale = 0.001

    vel_reward_w = 0.2
    avel_reward_w = 0.3
    rpos_reward_w = 3
    joints_reward_w = 2
    key_joints_reward_w = 4
    rot_reward_w = 1
    regulate_reward_w = 0.05

    # print("compute reward begin") #DEBUG ERRORSCALE
    rv_reward = _mse_reward(tar_rv, rv, vel_err_scale)
    rav_reward = _mse_reward(tar_rav, rav, avel_err_scale)
    rpos_reward = _mse_reward(tar_rpos, rpos, rpos_err_scale)

    tar_rot = torch.cat([tar_rrot, tar_dof_pos], dim=-1)
    rot = torch.cat([rrot, dof_pos], dim=-1)
    # rot_reward = _mse_reward(tar_rot, rot, rot_err_scale)

    rot_smpl = torch.zeros(list(rot.shape[:-1]) + [24, 3], device=rot.device)
    rot_smpl[..., DOF_BODY_IDS, :] = rot[..., 4:].view(list(rot.shape[:-1]) + [21, 3])
    rot_smpl[..., 0, :] = quat_to_exp_map(rot[:, :4])

    tar_rot_smpl = torch.zeros(list(tar_rot.shape[:-1]) + [24, 3], device=tar_rot.device)
    tar_rot_smpl[..., DOF_BODY_IDS, :] = tar_rot[..., 4:].view(list(tar_rot.shape[:-1]) + [21, 3])
    tar_rot_smpl[..., 0, :] = quat_to_exp_map(tar_rot[:, :4])

    # joints = body_model(global_orient=rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints
    joints = body_model(global_orient=tar_rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints #remove root rotation bias
    tar_joints = body_model(global_orient=tar_rot_smpl[..., :1, :], body_pose=tar_rot_smpl[..., 1:, :], return_verts=False).joints
    
    joints_reward = _mse_reward(tar_joints.view([tar_joints.shape[0], -1]), joints.view([tar_joints.shape[0], -1]), joints_err_scale)

    joints = body_model(global_orient=rot_smpl[..., :1, :], body_pose=rot_smpl[..., 1:, :], return_verts=False).joints #keep root rotation
    key_joints_reward = _mse_reward(tar_joints[:, key_joints_idx].view([tar_joints.shape[0], -1]), joints[:, key_joints_idx].view([tar_joints.shape[0], -1]), kjoints_err_scale)

    tar_dof_obs = dof_to_obs(tar_dof_pos)
    tar_root_rot_obs = quat_to_tan_norm_yup(tar_rrot)
    tar_rot = torch.cat([tar_root_rot_obs, tar_dof_obs], dim=-1)

    dof_obs = dof_to_obs(dof_pos)
    root_rot_obs = quat_to_tan_norm_yup(rrot)
    rot = torch.cat([root_rot_obs, dof_obs], dim=-1)
    rot_reward = _mse_reward(tar_rot, rot, rot_err_scale)

    dof_vel_regular = _mse_reward(dof_vel, torch.zeros_like(dof_vel), regulate_err_scale)

    # print("compute reward end") #DEBUG ERRORSCALE
    reward = vel_reward_w*rv_reward + avel_reward_w*rav_reward + rot_reward_w*rot_reward + rpos_reward_w*rpos_reward\
         + joints_reward_w*joints_reward + regulate_reward_w*dof_vel_regular + key_joints_reward_w*key_joints_reward#TODO weight
    reward /= vel_reward_w + avel_reward_w + rot_reward_w + rpos_reward_w + joints_reward_w + regulate_reward_w + key_joints_reward_w

    print("Cloning reward: ",torch.mean(reward))

    return reward
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_cloning_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs): #TODO
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    obs = torch.cat([root_states, dof_pos], dim=-1)
    return obs

@torch.jit.script
def _mse_reward(a, b, scale):
    # type: (Tensor, Tensor, float) -> Tensor
    err = a - b
    se = torch.sum(err*err, dim=-1)
    # print("reward",torch.mean(se))
    reward = torch.exp(-scale * se)
    return reward

@torch.jit.script
def compute_cloning_reset(reset_buf, motion_lengths, progress_buf, start_times, dt, \
    rigid_body_pos, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, bool, float) -> Tensor
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + start_times
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset

@torch.jit.script
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
    save_controllimit = 1e+10
    save_control_fail = (torch.abs(root_states)>save_controllimit).any(dim=-1)
    if save_control_fail.any():
        terminated = torch.where(save_control_fail, torch.ones_like(reset_buf), terminated)
    nan_error = torch.isnan(root_states).any(dim=-1)
    if nan_error.any():
        terminated = torch.where(nan_error, torch.ones_like(reset_buf), terminated)
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