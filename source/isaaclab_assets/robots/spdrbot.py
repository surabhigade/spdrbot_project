from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

## Configuration for the Spiderbot using a USD file
SPDRBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/surabhi/Downloads/V3urdfassembly_spdrbot_description (1)/spdrbot_video.usd", 
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
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],
            stiffness=40.0,
            damping=5.0,
            armature=0.01
        ),
    },
) 
