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
    # actions: 12 (Revolute joints)
    # observations: 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 12 (pos) + 12 (vel) + 12 (prev_actions) = 45
    action_space = 12       # 12 joints
    observation_space = 45  # 3+3+3+12+12+12
    state_space = 0         # Not used for PPO usually
    
    
    num_actions = 12
    num_observations = 45
    num_states = 0


    # Reward Scales
    rew_scale_forward_velocity: float = 1.0
    rew_scale_orthogonal_velocity: float = -0.05  # Missing attribute fixed
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

    # robot(s)
    # We replace the Cartpole config with your Spiderbot config
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
    

"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class SpdrbotProjectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    """