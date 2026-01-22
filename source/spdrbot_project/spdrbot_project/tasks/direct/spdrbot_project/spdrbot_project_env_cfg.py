# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import your custom spiderbot asset config
# Make sure the import path matches where you saved spdrbot.py
from isaaclab_assets.robots.spdrbot import SPDRBOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sim import GroundPlaneCfg
import isaaclab.sim as sim_utils

@configclass
class SpdrbotProjectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4        # Control frequency: 120Hz / 4 = 30Hz (Standard for RL)
    episode_length_s = 10.0 # Give the robot 15 seconds to try and walk
    
    # - spaces definition (Based on the list in env.py)
    action_space = 12       # 12 joints
    observation_space = 45  # 3+3+3+12+12+12
    state_space = 0         # Not used for PPO usually
    
    
    num_actions = 12
    num_observations = 45
    num_states = 0


    # Reward Scales
    rew_scale_forward_velocity: float = 1.0
    rew_scale_orthogonal_velocity: float = -0.05  
    rew_scale_angular_velocity_z: float = 0.05
    rew_scale_joint_deviation: float = -0.01
    rew_scale_upright: float = 1.0
    
    # Termination
    termination_height: float = 0.15 # Ensure this is also here

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, 
        render_interval=decimation,
        # Use GPU simulation for 4096 environments
        device="cuda",  
        use_fabric=True,
        #enable_scene_query_support=False,
    )

    # robot
    robot_cfg: ArticulationCfg = SPDRBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, 
        env_spacing=2.5, 
        replicate_physics=True
    )

    # We initialize it empty and then assign the path to avoid TypeError
    scene.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(), # This puts the spawner inside the 'spawn' attribute
    )
    # custom parameters/scales
    # - action scale: How much the AI's -1 to 1 signal moves the joints (in radians)
    action_scale = 0.25  
