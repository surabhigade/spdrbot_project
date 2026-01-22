# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from isaaclab.utils.math import quat_apply, quat_conjugate 
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .spdrbot_project_env_cfg import SpdrbotProjectEnvCfg


class SpdrbotProjectEnv(DirectRLEnv):
    cfg: SpdrbotProjectEnvCfg

    def __init__(self, cfg: SpdrbotProjectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get indices for all 12 joints (Revolute80 to Revolute91)
        self._joint_indices, _ = self.robot.find_joints("Revolute.*")
        
        # Buffers for robot data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        #self.root_pos = self.robot.data.root_link_pos
        #self.root_quat = self.robot.data.root_link_quat
        self.root_state = self.robot.data.root_state_w
        self.last_actions = torch.zeros(self.cfg.scene.num_envs, self.cfg.num_actions, device=self.device)

    def _setup_scene(self):
        # Spawn the robot defined in spdrbot.py config
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU if necessary
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Actions come from the RL policy (usually normalized -1 to 1)
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Convert normalized actions to joint position targets
        target_joint_pos = self.robot.data.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=self._joint_indices)

    def _get_observations(self) -> dict:
        # 1. Get the gravity vector in the world frame 
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        
        # 2. Extract quaternion from root_state_w [pos(0:3), quat(3:7), vel(7:10), ang_vel(10:13)]
        quat = self.robot.data.root_state_w[:, 3:7]
        
        # 3. Rotate the gravity vector into the robot's local body frame
        self.projected_gravity = quat_apply(quat_conjugate(quat), gravity_vec)

        # Standard locomotion observation vector
        obs = torch.cat(
            (
                self.robot.data.root_state_w[:, 7:10], # Linear velocity (3)
                self.robot.data.root_state_w[:, 10:13], # Body angular velocity (3)
                self.projected_gravity,       # Orientation relative to gravity (3)
                self.robot.data.joint_pos,               # Current joint positions (12)
                self.robot.data.joint_vel,               # Current joint velocities (12)
                self.actions,                            # Last action taken (12)
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:

        # 1. Transform World Velocity to Local Velocity 
        quat = self.robot.data.root_state_w[:, 3:7]
        lin_vel_w = self.robot.data.root_state_w[:, 7:10]
        local_lin_vel = quat_apply(quat_conjugate(quat), lin_vel_w)
        local_ang_vel = quat_apply(quat_conjugate(quat), self.robot.data.root_state_w[:, 10:13])

        # Compute rewards based on forward velocity and stability
        total_reward = compute_rewards(
            local_lin_vel=local_lin_vel,
            local_ang_vel=local_ang_vel,
            root_pos_z=self.robot.data.root_state_w[:, 2],      # Pass Z height
            joint_pos=self.robot.data.joint_pos,
            joint_vel=self.robot.data.joint_vel,
            projected_gravity=self.projected_gravity,
            actions=self.actions,
            last_actions=self.last_actions,
            reset_terminated=self.reset_terminated,
            forward_vel_scale=self.cfg.rew_scale_forward_velocity,
            upright_scale=self.cfg.rew_scale_upright,             
        )
        # Store current actions for the next step penalty
        self.last_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset if body is too low (collapsed) or flipped over
        quat = self.robot.data.root_state_w[:, 3:7]
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        proj_grav = quat_apply(quat_conjugate(quat), gravity_vec)
        
        # Termination conditions
        died = self.robot.data.root_state_w[:, 2] < self.cfg.termination_height
        flipped = proj_grav[:, 2] > 0.0

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (died | flipped), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joints to default standing pose with some noise
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, joint_pos.device)
        joint_vel = torch.zeros_like(joint_pos)

        # Reset base position to origin (with height offset)
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    local_lin_vel: torch.Tensor,
    local_ang_vel: torch.Tensor,
    root_pos_z: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    forward_vel_scale: float,
    upright_scale: float,
    target_height: float = 0.28,

    target_vel: float = 0.5,
    )-> torch.Tensor:

    # 1. Forward Velocity Reward (Target 0.5 m/s for larger steps)
    rew_vel = torch.exp(-torch.square(target_vel - local_lin_vel[:, 0]) / 0.25) * forward_vel_scale

    # 2. Joint Movement Reward: 
    # Encourage the legs to actually move (velocity) instead of staying static.
    # We scale this so it's not too high, or it will just shake again.
    rew_joint_move = torch.sum(torch.abs(joint_vel), dim=-1) * 0.01

    # 3. Upright Reward: Keep the body flat
    # projected_gravity[:, 2] is -1.0 when perfectly upright. 
    rew_upright = torch.exp(-torch.square(projected_gravity[:, 2] + 1.0) / 0.1) * upright_scale
    
    # 4. Height Reward (CRITICAL for Spider-look):
    # Penalty for being too high or too low. Keeps it in a crouched spider pose.
    rew_height = torch.exp(-torch.square(root_pos_z - target_height) / 0.01) * 2.0

    # 5. Stability Penalties (Yaw/Pitch/Roll rates)
    pen_ang_vel = torch.sum(torch.square(local_ang_vel), dim=-1) * -0.05
    pen_sideways = torch.square(local_lin_vel[:, 1]) * -1.0
    
    # 5. Loosen the Action Rate Penalty:
    # If this penalty is too high (e.g., -0.1), the robot is "scared" to move its joints fast.
    # Reduce it to -0.01 or -0.02.
    pen_action_rate = torch.sum(torch.square(actions - last_actions), dim=-1) * -0.02
    
    # 3. Torque/Energy Penalty (Requires joint torques if available, otherwise use action magnitude)
    pen_action_mag = torch.sum(torch.square(actions), dim=-1) * -0.01
    
     # 5. Action Smoothing: Penalize large/abrupt joint movements
    pen_action = torch.sum(torch.square(actions), dim=-1) * -0.01

    # Totaling
    total_reward = rew_vel + rew_joint_move + rew_upright + rew_height + pen_action_rate + pen_ang_vel + pen_action + pen_sideways + pen_action_mag
    
    # Mask rewards for dead robots
    total_reward =  torch.where(reset_terminated, torch.zeros_like(total_reward), total_reward)
    return total_reward
