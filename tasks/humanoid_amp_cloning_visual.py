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


from tasks import humanoid_amp_cloning
from tasks.humanoid_amp_cloning import NUM_MOTION_TAR_PER_STEP
from tasks.mytask.utils_amp.motion_lib import MotionLib
from tasks.mytask.humanoid_amp_base import DOF_BODY_IDS, EXTERNAL_IDX, EXTERNAL_ACTIONS, DOF_ACTIONS

from isaacgym.torch_utils import *
from utils.torch_jit_utils import *

class HumanoidCloning_v(humanoid_amp_cloning.HumanoidCloning):
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
        num_actors = self.get_num_actors_per_env()
        self._humanoid_actor_ids_ref = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1
        self._humanoid_root_states_ref = self._root_states.view(self.num_envs, num_actors, 13)[..., 1, :]
        self.dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos_ref = self._dof_state.view(self.num_envs, self.dofs_per_env, 2)[..., self.num_dof:, 0]
        self._dof_vel_ref = self._dof_state.view(self.num_envs, self.dofs_per_env, 2)[..., self.num_dof:, 1]
        return
    
    def _build_env(self, env_id, env_ptr):
        super()._build_env(env_id, env_ptr)
        self._build_ref_humanoid(env_id, env_ptr)
        return
    
    def _build_ref_humanoid(self, env_id, env_ptr):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/urdf')
        asset_file = "Humanoid/urdf/Humanoid.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
        
        asset_options = gymapi.AssetOptions()
        asset_options.enable_gyroscopic_forces = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity=True
        asset_options.angular_damping = 100
        asset_options.max_angular_velocity = 100.0 #100.0 consistant with prop['velocity']
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE #lets the joints move freely within their range of motion
        asset_options.density = 985
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.94, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid_ref", env_id+self.num_envs, 1, 0)
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.94, 1., 0.94))
        
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions[..., :DOF_ACTIONS])
            dof_pos = self._dof_state.view(self.num_envs, self.dofs_per_env, 2)[..., 0]
            dof_pos[..., :self.num_dof] = pd_tar
            pd_tar = dof_pos.contiguous()
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor) #apply the driven 63
        else:
            forces = self.actions[..., :DOF_ACTIONS] * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        motor_scale = self.body_effort * self.power_scale
        forces_all = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float)
        forces = forces_all[:, :self.num_bodies]
        forces[:, self._rigid_shape_ids[EXTERNAL_IDX]] = self.actions[..., DOF_ACTIONS:].view(self.num_envs, len(EXTERNAL_IDX), 3)      
        forces *= motor_scale
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces_all))

        self._update_task_pre()
        
        forces = torch.zeros_like(self.actions[..., :DOF_ACTIONS])
        forces = torch.cat([forces, forces], dim=-1) #for full actor
        force_tensor = gymtorch.unwrap_tensor(forces)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        env_ids_int32 = self._humanoid_actor_ids_ref[env_ids]
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim, force_tensor, gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    # def _update_task_pre(self):
    #     super()._update_task_pre()
    #     forces = torch.zeros_like(self.actions)
    #     forces = torch.cat([forces, forces], dim=-1) #for full actor
    #     force_tensor = gymtorch.unwrap_tensor(forces)
    #     env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    #     env_ids_int32 = self._humanoid_actor_ids_ref[env_ids]
    #     self.gym.set_dof_actuation_force_tensor_indexed(self.sim, force_tensor, gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     return

        return
    def _update_task_post(self):
        super()._update_task_post()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        tar_root_pos = self._motion_tar_buf[:, :3].clone()
        root_height_offset = 0.05 #see prepare_motion #273
        tar_root_pos[..., 1] -= root_height_offset #offset
        tar_root_rot = self._motion_tar_buf[:, 3:7].clone()
        tar_root_vel = torch.zeros_like(self._motion_tar_buf[:, 7:10])
        tar_root_ang_vel = torch.zeros_like(self._motion_tar_buf[:, 10:13])
        tar_dof_pos = self._motion_tar_buf[:, 13: 76].clone()
        tar_dof_vel = torch.zeros_like(self._motion_tar_buf[:, 13: 76])
        self._set_env_state_ref(env_ids, tar_root_pos, tar_root_rot, tar_dof_pos, tar_root_vel, tar_root_ang_vel, tar_dof_vel)

        return

    def _set_env_state_ref(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states_ref[env_ids, 0:3] = root_pos
        self._humanoid_root_states_ref[env_ids, 3:7] = root_rot
        self._humanoid_root_states_ref[env_ids, 7:10] = root_vel
        self._humanoid_root_states_ref[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos_ref[env_ids] = dof_pos
        self._dof_vel_ref[env_ids] = dof_vel

        env_ids_int32 = self._humanoid_actor_ids_ref[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return


