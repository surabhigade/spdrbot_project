"""
SpiderBot Robot Asset Configuration for Isaac Lab

This module defines the configuration for the SpiderBot 12-legged robot used in the
reinforcement learning environment. The robot is loaded from a USD file with implicit
actuators configured for joint position control.

Author: SpiderBot RL Project
"""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg


# Configuration for the SpiderBot using a USD file
SPDRBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/surabhi/Downloads/V3urdfassembly_spdrbot_description (1)/spdrbot_video.usd",
        # TODO: Consider making this path configurable or relative
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Initial position (x, y, z)
        joint_pos={".*": 0.0},  # Initialize all joints to 0 radians
    ),
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],  # Match all Revolute joints (12 joints)
            stiffness=40.0,                   # Joint stiffness (N·m/rad)
            damping=5.0,                      # Joint damping (N·m·s/rad)
            armature=0.01,                    # Motor armature inertia
        ),
    },
)
