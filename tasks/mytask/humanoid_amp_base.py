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


import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils.torch_jit_utils import *
from ..base.vec_task import VecTask

DOF_BODY_IDS = [3, 6, 9, 13, 16, 18, 20, 12, 15, 14, 17, 19, 21, 2, 5, 8, 11, 1, 4, 7, 10] #index: urdf joints; value: motionlib joints. engine2dataset
# DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
DOF_OFFSETS = list(range(0, 64, 3))
# NUM_OBS = 13 + 52 + 28 + 12 # [(root_h, root_rot, root_vel, root_ang_vel), dof_pos, dof_vel, key_body_pos 3*4]
# NUM_OBS = 1 + 22 * (3 + 6 + 3 + 3) - 3
NUM_OBS = 1 + 22 * (3 + 6 + 3 + 3) - 3 + 63 #residual control
# NUM_ACTIONS = 28
DOF_ACTIONS = 63
EXTERNAL_IDX = [0]
EXTERNAL_ACTIONS = len(EXTERNAL_IDX)*3
NUM_ACTIONS = DOF_ACTIONS + EXTERNAL_ACTIONS
SHAPES = {  
            'hip': 0, #0b10011, 19
            'spine': 1, 'spine1': 2, 'spine2': 7, 'neck': 8, 'head': 9, #0b10000, 16
            'left_shoulder': 3, #0b11111, 31
            'left_arm': 4, 'left_fore_arm': 5, 'left_hand': 6, #0b01000, 8
            'right_shoulder': 10, #0b11111, 31
            'right_arm': 11, 'right_fore_arm': 12, 'right_hand': 13, #0b00100, 4
            'right_upleg': 14, 'right_leg': 15, 'right_foot': 16, 'right_toe': 17, #0b00010, 2
            'left_upleg': 18, 'left_leg': 19, 'left_foot': 20, 'left_toe': 21, #0b00001, 1
            }#in DFS order(physx rigid shape order)


KEY_BODY_NAMES = ["right_hand", "left_hand", "right_toe", "left_toe", 'head']

RIGID_SHAPE_NAMES = [
                    'hip_rx',

                    'spine_rx',
                    'spine1_rx',
                    'spine2_rx',
                    'neck_rx',
                    'head',

                    'left_shoulder_rx',
                    'left_arm_rx',
                    'left_fore_arm_rx',
                    'left_hand',

                    'right_shoulder_rx',
                    'right_arm_rx',
                    'right_fore_arm_rx',
                    'right_hand',

                    'left_UpLeg_rx',
                    'left_leg_rx',
                    'left_foot_rx',
                    'left_toe',
                    
                    'right_UpLeg_rx',
                    'right_leg_rx',
                    'right_foot_rx',
                    'right_toe',         
                    ]

class HumanoidAMPBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        '''
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}
        '''
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        '''
        Retrieves buffer for Actor root states. The buffer has shape (num_actors, 13). 
        State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        '''
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) 
        '''
        (self.num_envs/num_actors * self.num_dof, 2): (4096*63, 2); Each DOF state contains position and velocity.
        '''
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim) #foot contact force; for detail, search: create force sensor
        '''
        The buffer has shape (num_force_sensors, 6). Each force sensor state has forces (3) and torques (3) data.
        '''
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        '''
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        '''
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        '''
        The buffer has shape (num_rigid_bodies, 3). Each contact force state contains one value for each X, Y, Z axis.
        '''

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) #TODO
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, -1)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        # print('num actor:', num_actors)
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices, for reset
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # print('dof state shape: ', self._dof_state.shape)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        # print('dof per env:', dofs_per_env)
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_z_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "17_rz")
        left_shoulder_z_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "16_rz")
        # print(right_shoulder_z_handle)
        self._initial_dof_pos[:, right_shoulder_z_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_z_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # print('body per env:', bodies_per_env)
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        if self.viewer != None:
            self._init_camera()
        
        # print(self.progress_buf[0])
        # for i in range(self.num_envs):
        #     self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._initial_dof_pos))
        #     # self._reset_actors(torch.LongTensor([i]).to(self.device))
        #     break
        # self._refresh_sim_tensors()
        # print('refresh:', self._dof_pos[i, 14])

        print('INI SUCCEED')
        return
    def _build_env(self, env_id, env_ptr):
        pass

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors
        
    def create_sim(self):
        # self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.up_axis_idx = 1
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        # if torch.isnan(self._humanoid_root_states).any():
        #     print('root_states reset')
        #     for i in env_ids:
        #         if torch.isnan(self._humanoid_root_states[i]).any():
        #             print(self._humanoid_root_states[i])
        #     raise
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col, env_ids=None):
        if (env_ids is None):
            env_ids = range(self.num_envs)       
        for i in env_ids:
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/urdf')
        asset_file = "Humanoid/urdf/Humanoid.urdf"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 400.0 #100.0 consistant with prop['velocity']
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE #lets the joints move freely within their range of motion
        asset_options.density = 985 #2200 #985
        # asset_options.override_com = True
        # asset_options.override_inertia = True #yes: lead to abnormal pose balance, why? Don't setting <inertial> in urdf; no: lead to continueslly rolling, since inertia are all zero in our urdf
        # asset_options.fix_base_link = True


        # asset_options.collapse_fixed_joints = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        print('LOAD ASSETS SUCCEED')

        # print('Num of Dof: ', self.gym.get_asset_dof_count(humanoid_asset), self.gym.get_asset_actuator_count(humanoid_asset))
        # actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        # motor_efforts = [prop.motor_effort for prop in actuator_props] #urdf doesn't support actuator
        self.dof_props = self.gym.get_asset_dof_properties(humanoid_asset)
        '''
        dof prorerties
        hasLimits - Flags whether the DOF has limits.
        lower - lower limit of DOF. In radians or meters
        upper - upper limit of DOF. In radians or meters
        driveMode - Drive mode for the DOF. See GymDofDriveMode.
        velocity - Maximum velocity of DOF. In Radians/s, or m/s
        effort - Maximum effort of DOF. in N or Nm.
        stiffness - DOF stiffness.
        damping - DOF damping.
        friction - DOF friction coefficient, a generalized friction force is calculated as DOF force multiplied by friction.
        armature - DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia. Larger values could improve simulation stability.
        '''
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        # print(dof_props)
        for name, prop in zip(self.dof_names, self.dof_props):
            stiffness = self.cfg['env']['joint_params'][name][0]
            prop['velocity'] = 400 #100 IMPORTANT inconsistent will cause nan
            prop['effort'] = self.cfg['env']['joint_params'][name][-1]
            prop['stiffness'] = stiffness
            prop['damping'] = self.cfg['env']['joint_params'][name][1]
            prop['armature'] = stiffness/10000
            # print(prop)
            # print(prop['driveMode'])
        # print(self.dof_props)
        motor_efforts = [prop['effort'] for prop in self.dof_props] #for temporary use

        # resolve self-collision setting
        shape_props = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        for idx, shape_prop in enumerate(shape_props):
            shape_prop.rolling_friction = 0.05
            shape_prop.torsion_friction = 0.05
            # print('restoffset:', shape_prop.rest_offset, shape_prop.contact_offset)
            # shape_prop.rest_offset = -0.05
            # print('shape restitution:', shape_prop.restitution)
            if idx in [0, 14, 18]:
                shape_prop.filter = 19
            elif idx in [1, 2, 7, 8, 9]:
                shape_prop.filter = 16
            elif idx in [5, 6]:
                shape_prop.filter = 8
            elif idx in [12, 13]:
                shape_prop.filter = 4
            elif idx in [15, 16, 17]:
                shape_prop.filter = 2
            elif idx in [19, 20, 21]:
                shape_prop.filter = 1
            else:
                shape_prop.filter = 31
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, shape_props)

        # shape_props = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        # for idx, shape_prop in enumerate(shape_props):
        #     print('shape restitution:', shape_prop.restitution)
        #     print('shape filter:', shape_prop.filter)
        #     print('shape friction:', shape_prop.friction, shape_prop.compliance, shape_prop.rolling_friction, shape_prop.torsion_friction)
        
            
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_toe")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_toe")
        sensor_pose = gymapi.Transform()

        #create force sensor
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.names_dof = self.gym.get_asset_dof_names(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
        print('Num of Dof: ', self.num_dof)
        print('Num of bodies: ', self.num_bodies)
        print('Num of joints: ', self.num_joints)
        print('Num of shapes:', self.num_shapes)
        # print(self.gym.get_asset_soft_body_count(humanoid_asset)) #0
        # print(self.gym.get_asset_soft_materials(humanoid_asset)) #[]
        # # Iterate through bodies the order is base on Depth-First-Search for the tree, for visulization see pdf
        # print("Bodies:")
        # for i in range(self.num_bodies):
        #     name = self.gym.get_asset_rigid_body_name(humanoid_asset, i)
        #     print(" %2d: '%s'" % (i, name))
        # print(self.gym.get_asset_rigid_body_names(humanoid_asset))
        # print(self.gym.get_asset_dof_names(humanoid_asset)) #this would be useful to map the order
        # print(len(indices))
        # for indice in indices:
        #     print(indice.start)
        # # Iterate through joints
        # print("Joints:")
        # for i in range(self.num_joints):
        #     name = self.gym.get_asset_joint_name(humanoid_asset, i)
        #     type = self.gym.get_asset_joint_type(humanoid_asset, i)
        #     type_name = self.gym.get_joint_type_string(type)
        #     print(" %2d: '%s' (%s)" % (i, name, type_name))

        # # iterate through degrees of freedom (DOFs)
        # print("DOFs:")
        # for i in range(self.num_dof):
        #     name = self.gym.get_asset_dof_name(humanoid_asset, i)
        #     type = self.gym.get_asset_dof_type(humanoid_asset, i)
        #     type_name = self.gym.get_dof_type_string(type)
        #     print(" %2d: '%s' (%s)" % (i, name, type_name))

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.94, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.body_effort = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # contact_filter = 0 #all collision
            contact_filter = -1 #using asset setting
            # contact_filter = 1 #no collision
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self._build_env(i, env_ptr)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))
            # self._build_env(i, env_ptr)
            
            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            self.body_effort.append([body_prop.mass * 10 for body_prop in body_props])
            # for idx, body_prop in enumerate(body_props):
            #     print('body:', body_prop.com, body_prop.flags, body_prop.inertia)

            if (self._pd_control):
                # dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop = self.dof_props.copy()
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS #set drive mode 
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        self.body_effort = to_torch(self.body_effort, device=self.device)[..., None].expand(-1, -1, 3)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._rigid_shape_ids = self._build_rigid_shape_ids_tensor(self.envs[0], self.humanoid_handles[0])
        self._key_body_ids = self._build_key_body_ids_tensor(self.envs[0], self.humanoid_handles[0])
        self._contact_body_ids = self._build_contact_body_ids_tensor(self.envs[0], self.humanoid_handles[0])
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3): 
                # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi
                
                ######alpha version
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale #alpha version, may cause nan error while ase use it, might because of joint setting in urdf
                #########

                # curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                # curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                # curr_mid = 0.5 * (curr_high + curr_low)
                
                # # extend the action range to be a bit beyond the joint limits so that the motors
                # # don't lose their strength as they approach the joint limits
                # curr_scale = 0.7 * (curr_high - curr_low)
                # curr_low = curr_mid - curr_scale
                # curr_high = curr_mid + curr_scale

                # lim_low[dof_offset:(dof_offset + dof_size)] = curr_low
                # lim_high[dof_offset:(dof_offset + dof_size)] = curr_high

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) + 0.01 * compute_regular_reward(actions[..., DOF_ACTIONS:])
        if torch.isnan(self.rew_buf).any():
            self.rew_buf[:] = torch.where(torch.isnan(self.rew_buf), torch.zeros_like(self.rew_buf), self.rew_buf)
        
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self._humanoid_root_states, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # if torch.isnan(self._root_states).any():
        #     self._root_states[:] = torch.where(torch.isnan(self._root_states), torch.ones_like(self._root_states)*1e+15, self._root_states)
        # if torch.isnan(self._dof_state).any():
        #     self._dof_state[:] = torch.where(torch.isnan(self._dof_state), torch.ones_like(self._dof_state)*1e+15, self._dof_state)
        # if torch.isnan(self._rigid_body_state).any():
        #     self._rigid_body_state[:] = torch.where(torch.isnan(self._rigid_body_state), torch.ones_like(self._rigid_body_state)*1e+15, self._rigid_body_state)
        
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
            
        if torch.isnan(self.obs_buf).any():  #safe check, but will slow down the process
            self.obs_buf[:] = torch.where(torch.isnan(self.obs_buf), torch.zeros_like(self.obs_buf), self.obs_buf)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        # if (env_ids is None):
        #     root_states = self._humanoid_root_states
        #     dof_pos = self._dof_pos
        #     dof_vel = self._dof_vel
        #     key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        # else:
        #     root_states = self._humanoid_root_states[env_ids]
        #     dof_pos = self._dof_pos[env_ids]
        #     dof_vel = self._dof_vel[env_ids]
        #     key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        # if torch.isnan(root_states).any():
        #     print('root_states')
        #     for root_state in root_states:
        #         if torch.isnan(root_state).any():
        #             print(root_state)
        #     raise
        # obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
        #                                     key_body_pos, self._local_root_obs)

        if (env_ids is None):
            body_pos = self._rigid_body_pos[..., self._rigid_shape_ids, :]
            body_rot = self._rigid_body_rot[..., self._rigid_shape_ids, :]
            body_vel = self._rigid_body_vel[..., self._rigid_shape_ids, :]
            body_ang_vel = self._rigid_body_ang_vel[..., self._rigid_shape_ids, :]
            dof_pos = self._dof_pos
        else:
            body_pos = self._rigid_body_pos[env_ids][..., self._rigid_shape_ids, :]
            body_rot = self._rigid_body_rot[env_ids][..., self._rigid_shape_ids, :]
            body_vel = self._rigid_body_vel[env_ids][..., self._rigid_shape_ids, :]
            body_ang_vel = self._rigid_body_ang_vel[env_ids][..., self._rigid_shape_ids, :]
            dof_pos = self._dof_pos[env_ids]
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs)
        #TODO Try compute_humanoid_observations_max with rigid body pose
        obs = torch.cat([obs, dof_pos], dim=-1) #residual control
        # if torch.isnan(obs).any():
        #     print('obs nan')
        #     for i in range(obs.shape[0]):
        #         if torch.isnan(obs[i]).any():
        #             print(obs[i])
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        res1 = self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        res2 = self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # print(res1, res2)
        # print(self._dof_pos[env_ids, 14])
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):#TODO
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions[..., :DOF_ACTIONS])
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
        # print(forces[0, self._rigid_shape_ids[0]])
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces_all))

        return

    def post_physics_step(self):#TODO
        self.progress_buf += 1

        self._refresh_sim_tensors()
        # if torch.isnan(self._humanoid_root_states).any():
        #     print('root_states')
        #     for i in range(len(self._humanoid_root_states)):
        #         if torch.isnan(self._humanoid_root_states[i]).any():
        #             print(self._humanoid_root_states[i])
        #     raise
        # if torch.isnan(self._dof_state).any():
        #     print('_dof_state')
        #     for i in range(len(self._dof_state)):
        #         if torch.isnan(self._dof_state[i]).any():
        #             print(self._dof_state[i])
        #     raise
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, mode="rgb_array"):
        if self.viewer and self.camera_follow:
            self._update_camera()

        rt = super().render(mode)
        return rt

    def _build_rigid_shape_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in RIGID_SHAPE_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action): #a trick? so here 'action' is far from state. (-1, 1)
        pd_tar = self._pd_action_offset + self._pd_action_scale * action #work
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].detach().cpu().numpy()
        # print(self._cam_prev_char_pos)
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0] - 3.0, 
                              1.0,
                              self._cam_prev_char_pos[2] - 3.0, 
                              )
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 1.0,
                                 self._cam_prev_char_pos[2],
                                 )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print(self._humanoid_root_states[0, 0:3].cpu().numpy(),self._cam_prev_char_pos)
        char_root_pos = self._humanoid_root_states[0, 0:3].detach().cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], 1.0, char_root_pos[2],)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  cam_pos[1],
                                  char_root_pos[2] + cam_delta[2], 
                                  )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    # dof_obs_size = 52
    # dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    dof_obs_size = 63 * 2
    dof_offsets = list(range(0, 64, 3))
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3): #should it be euler to quat?? ###NOTE IMPORTANT AND WEIRD
            joint_pose_q = exp_map_to_quat(joint_pose) #BUG ?
            # joint_pose = torch.where(joint_pose<0, joint_pose+2*np.pi, joint_pose)
            # joint_pose_q = quat_from_euler_xyz(joint_pose[..., 0], joint_pose[..., 1], joint_pose[..., 2])

            joint_dof_obs = quat_to_tan_norm_yup(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    # if torch.isnan(root_states).any():
    #     print('root_states')
    # if torch.isnan(dof_vel).any():
    #     print('dof_vel')
    # if torch.isnan(key_body_pos).any():
    #     print('key_body_pos')
    # if torch.isnan(dof_pos).any():
    #     print('dof_pos')
    # print('root_states', torch.isnan(root_states).any())
    # print('dof_vel', torch.isnan(dof_vel).any())
    # print('key_body_pos', torch.isnan(key_body_pos).any())

    # for idx, i in enumerate(root_states):
    #     if torch.isnan(i).any():
    #         print(i)
    #         print(idx)
    # for idx, i in enumerate(dof_vel):
    #     if torch.isnan(i).any():
    #         print(i)
    #         print(idx)
    # for idx, i in enumerate(key_body_pos):
    #     if torch.isnan(i).any():
    #         print(i)
    #         print(idx)
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 1:2]
    heading_rot = calc_heading_quat_inv_yup(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm_yup(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 1:2]
    heading_rot = calc_heading_quat_inv_yup(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm_yup(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_to_tan_norm_yup(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_reward(obs_buf): #just a space holder
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

def compute_regular_reward(actions): 
    # type: (Tensor) -> Tensor
    actions_scale = 10
    forces = torch.sum(actions * actions, dim=-1)
    reward = torch.exp(-actions_scale * forces)
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, root_states,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    save_controllimit = 1e+10
    save_control_fail = (torch.abs(root_states)>save_controllimit).any(dim=-1)
    if save_control_fail.any():
        terminated = torch.where(save_control_fail, torch.ones_like(reset_buf), terminated)
    nan_error = torch.isnan(root_states).any(dim=-1)
    if nan_error.any():
        terminated = torch.where(nan_error, torch.ones_like(reset_buf), terminated)
    if (enable_early_termination):
        #TODO
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        # body_height = rigid_body_pos[..., 1]
        # fall_height = body_height < termination_height
        # fall_height[:, contact_body_ids] = False
        # fall_height = torch.any(fall_height, dim=-1)

        # has_fallen = torch.logical_and(fall_contact, fall_height)
        body_height = rigid_body_pos[..., 0, 1] #ROOT
        has_fallen = body_height < termination_height

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1) #1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        flied = body_height > 2.5
        terminated = torch.where(flied, torch.ones_like(reset_buf), terminated)

    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


