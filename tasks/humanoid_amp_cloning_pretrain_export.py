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


from tasks import humanoid_amp_cloning_pretrain
from tasks.humanoid_amp_cloning_pretrain import NUM_MOTION_TAR_PER_STEP
from tasks.mytask.utils_amp.motion_lib import MotionLib
from tasks.mytask.humanoid_amp_base import DOF_BODY_IDS, EXTERNAL_IDX, EXTERNAL_ACTIONS, DOF_ACTIONS
from tasks.humanoid_ampmy import HumanoidAMPmy

from isaacgym.torch_utils import *
from utils.torch_jit_utils import *

import datetime

class HumanoidCloning_ampexport(humanoid_amp_cloning_pretrain.HumanoidCloning_pretrain):
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
        self._motion_ids -= self.num_envs

        #never go to reset actor
        self.max_episode_length = np.inf
        self._enable_early_termination = False
        return


    def post_physics_step(self): #NOTE
        super().post_physics_step() 
        #save obs for experiment
        obs_save = self.obs_buf.detach().cpu().numpy()[0]
        obs_save = torch.cat((self._humanoid_root_states[:, :7], self._dof_pos[:]), dim=-1) #[num_env, 3 + 4 + 21*3]
        obs_save = obs_save.detach().cpu().numpy()
        
        motion_ids = self._motion_ids.detach().cpu().numpy()
        for idx, motion_id in enumerate(motion_ids):
            name = self._motion_lib._motion_files[motion_id]
            name = os.path.split(name)[-1]
            name = name[:name.rfind('.')]
            time_str = (self.progress_buf[idx].detach() - self.cloningprogress_buf[idx].detach()).cpu() #*self.dt + self._start_times[idx]
            # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            obs_folder = os.path.abspath("obs/"+name)
            # Create output folder if needed
            os.makedirs(obs_folder, exist_ok=True)
            # savepath = os.path.join(obs_folder, time_str+".npy")
            savepath = os.path.join(obs_folder, "%06.npy" %time_str)
            np.save(savepath, obs_save[idx])

        return

    def _reset_task(self, env_ids):
        if self._state_init == HumanoidAMPmy.StateInit.Start:
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        
        self._init_amp_obs(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        self.cloningprogress_buf[env_ids] = self.progress_buf[env_ids].clone()

        return

    def _reset_actors(self, env_ids): #the function handle of reset, from public function reset_idx(self, env_ids) see above
        pass

        return

    def _reset_ref_state_init(self, env_ids):
        #TODO currently, only do with num_envs=1
        #DONE better remove the last motion by adding spaceholder with Tpose
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_ids[env_ids].detach().cpu().numpy() + self.num_envs
        motion_ids.clip(a_max=self._motion_lib.num_motions()-1)
            
        assert((motion_ids < self._motion_lib.num_motions()).any()), "Inference Done !!!!"
        
        if self._state_init == HumanoidAMPmy.StateInit.Start:
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

