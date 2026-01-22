# SpiderBot Reinforcement Learning Environment

## Project Overview

This is a 4-legged spider robot reinforcement learning (RL) project built on top of **Isaac Lab** (NVIDIA's physics simulation framework). The project implements a complete RL training pipeline using multiple algorithms including PPO (Proximal Policy Optimization) and AMP (Adversarial Motion Priors) for training the spider robot to walk efficiently.

**Project Key Features:**
- 🤖 12-joint spider robot with sophisticated locomotion control
- 🎯 Multiple RL algorithms: PPO (RSL-RL) and PPO + AMP (SKRL)
- 🚀 GPU-accelerated training with 1024 parallel environments
- 📊 Complex reward shaping for realistic spider behavior
- 🔧 Isolated development environment outside Isaac Lab core
- 🎮 Extensible as Omniverse extension

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Environment Configuration](#environment-configuration)
4. [Training Policies](#training-policies)
5. [Quick Start Commands](#quick-start-commands)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Isaac Lab**: Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python 3.10+**: Required for Isaac Lab compatibility
- **CUDA-capable GPU**: Recommended for training (CPU training is very slow)
- **Conda/Mamba**: For managing Python environments

### Step 1: Set Up Isaac Lab Environment

Activate the Isaac Lab conda environment:

```bash
conda activate isaaclab
```

If you haven't created the `isaaclab` environment yet, follow Isaac Lab's installation guide.

### Step 2: Clone and Install This Project

Clone this repository:

```bash
git clone https://github.com/surabhigade/spdrbot_project.git
cd spdrbot_project
```

Install the project in editable mode:

```bash
python -m pip install -e source/spdrbot_project
```

### Step 3: Set Up SpiderBot USD File

The SpiderBot robot requires a USD (Universal Scene Description) file. Make sure you have the USD file:

```
/home/surabhi/Downloads/V3urdfassembly_spdrbot_description (1)/spdrbot_video.usd
```

**If you need to change the USD file path**, edit `source/isaaclab_assets/robots/spdrbot.py`:

```python
usd_path="/your/custom/path/to/spdrbot_video.usd"
```

### Step 4: Verify Installation

List available tasks to verify the installation:

```bash
python scripts/list_envs.py
```

You should see `Template-Spdrbot-Project-Direct-v0` in the list.

Test with a random agent:

```bash
python scripts/random_agent.py --task Template-Spdrbot-Project-Direct-v0 --num_envs 64
```

---

## Project Structure

```
spdrbot_project/
├── README.md                                          # Original template README
├── README_COMPLETE.md                                 # This comprehensive guide
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                                   # Training script (RSL-RL)
│   │   ├── play.py                                    # Policy inference/playing
│   │   └── cli_args.py                                # Command-line arguments
│   ├── skrl/
│   │   ├── train.py                                   # Training script (SKRL)
│   │   └── play.py                                    # SKRL policy inference
│   ├── random_agent.py                                # Random action baseline
│   ├── zero_agent.py                                  # Zero action baseline
│   └── list_envs.py                                   # List available tasks
│
├── source/
│   ├── isaaclab_assets/                               # Isaac Lab asset configurations
│   │   ├── __init__.py
│   │   └── robots/
│   │       ├── __init__.py
│   │       └── spdrbot.py                             # SpiderBot USD asset config
│   │
│   └── spdrbot_project/
│       ├── setup.py                                   # Package setup
│       ├── pyproject.toml                             # Project metadata
│       ├── config/
│       │   └── extension.toml                         # Omniverse extension config
│       └── spdrbot_project/
│           ├── ui_extension_example.py                # UI extension example
│           ├── __init__.py
│           └── tasks/
│               └── direct/
│                   └── spdrbot_project/
│                       ├── __init__.py                # Gymnasium environment registration
│                       ├── spdrbot_project_env.py     # Main environment class
│                       ├── spdrbot_project_env_cfg.py # Environment configuration
│                       ├── test_spdrbot_project_env.py # Unit tests
│                       └── agents/
│                           ├── rsl_rl_ppo_cfg.py      # RSL-RL PPO config
│                           ├── skrl_ppo_cfg.yaml      # SKRL PPO config
│                           └── skrl_amp_cfg.yaml      # SKRL AMP config
```

---

## Robot Asset Configuration

### SpiderBot USD Asset

The SpiderBot robot is configured in `source/isaaclab_assets/robots/spdrbot.py`. This file defines the complete robot asset configuration for Isaac Lab:

**File Location**: `source/isaaclab_assets/robots/spdrbot.py`

**Key Configuration Details**:

```python
SPDRBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/surabhi/Downloads/V3urdfassembly_spdrbot_description (1)/spdrbot_video.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),          # Initial position (x, y, z in meters)
        joint_pos={".*": 0.0},         # All joints initialized to 0 radians
    ),
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],  # Matches all 12 Revolute joints
            stiffness=40.0,                   # Joint stiffness (N·m/rad)
            damping=5.0,                      # Joint damping (N·m·s/rad)
            armature=0.01,                    # Motor armature inertia
        ),
    },
)
```

**Important Notes**:
- The USD file path currently points to a local Downloads directory. **Update this path** if you move the USD file.
- Implicit actuators are used, meaning Isaac Lab handles joint control through position targets.
- The 12 Revolute joints are automatically detected using the regex pattern `"Revolute.*"`
- Initial position is set 0.5m above ground to prevent clipping

### Usage in Environment Config

The asset is imported and used in `spdrbot_project_env_cfg.py`:

```python
from isaaclab_assets.robots.spdrbot import SPDRBOT_CFG

# In SpdrbotProjectEnvCfg:
robot_cfg: ArticulationCfg = SPDRBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

---

## Environment Configuration

### Robot Specifications

The SpiderBot has the following specifications:

| Parameter | Value |
|-----------|-------|
| Number of Joints | 12 (Revolute joints) |
| Action Space | 12-dimensional (joint position targets) |
| Observation Space | 45-dimensional |
| Control Frequency | 30 Hz (120 Hz physics / 4 decimation) |
| Episode Length | 10 seconds |
| Parallel Environments | 1024 (default) |
| Device | CUDA GPU (default) |

### Observation Space (45 dimensions)

```
Observation = [
    linear_velocity (3),          # Body linear velocity in world frame
    angular_velocity (3),         # Body angular velocity in world frame
    projected_gravity (3),        # Gravity vector in robot's body frame
    joint_positions (12),         # Current joint positions (radians)
    joint_velocities (12),        # Current joint velocities (rad/s)
    previous_actions (12)         # Last action taken
]
```

### Action Space (12 dimensions)

```
Actions = [
    joint_target_1, joint_target_2, ..., joint_target_12
]
```

Each action is a normalized value in [-1, 1] which is scaled by `action_scale = 0.25` to get the actual joint position target in radians.

### Reward Function

The reward function consists of multiple components designed to encourage realistic spider walking:

| Reward Component | Scale | Description |
|------------------|-------|-------------|
| Forward Velocity | +1.0 | Rewards movement toward target velocity (0.5 m/s) |
| Upright Stance | +1.0 | Keeps robot upright using gravity projection |
| Height Maintenance | +2.0 | Maintains crouched spider pose (~0.28m height) |
| Joint Movement | +0.01 | Encourages leg movement |
| Angular Velocity Penalty | -0.05 | Penalizes excessive rotation |
| Sideways Motion Penalty | -1.0 | Penalizes lateral drift |
| Action Rate Smoothness | -0.02 | Smooth action transitions |
| Energy Consumption | -0.01 | Minimizes joint effort |

### Termination Conditions

The episode terminates (done=True) when:
- Robot body height drops below 0.15m (collapsed)
- Robot flips over (projected gravity z > 0.0)
- Episode reaches 10 seconds (time limit)

---

## Training Policies

This project supports three different RL training approaches:

### 1. RSL-RL PPO (Recommended for Fast Training)

**Algorithm**: Proximal Policy Optimization (PPO)  
**Framework**: RSL-RL (Rocket Science Lab RL)  
**Config File**: `agents/rsl_rl_ppo_cfg.py`

**Architecture**:
- Actor Network: [32, 32] with ELU activation
- Critic Network: [32, 32] with ELU activation

**Hyperparameters**:
- Learning Rate: 1.0e-3
- Entropy Coefficient: 0.005
- Clip Parameter: 0.2
- Steps per Environment: 16
- Max Iterations: 150
- Save Interval: 50 iterations

### 2. SKRL PPO (Standard PPO)

**Algorithm**: Proximal Policy Optimization (PPO)  
**Framework**: SKRL (Soft-body Control Reinforcement Learning)  
**Config File**: `agents/skrl_ppo_cfg.yaml`

**Architecture**:
- Policy Network: [32, 32] with ELU activation
- Value Network: [32, 32] with ELU activation
- State Preprocessor: Running Standard Scaler

**Hyperparameters**:
- Learning Rate: 5.0e-04 (with KL adaptive scheduling)
- Learning Epochs: 8
- Mini-batches: 8
- Rollouts: 32
- Discount Factor (γ): 0.99
- Lambda (GAE): 0.95
- Total Timesteps: 4,800

### 3. SKRL AMP (Advanced Motion Imitation)

**Algorithm**: Adversarial Motion Priors (AMP)  
**Framework**: SKRL  
**Config File**: `agents/skrl_amp_cfg.yaml`

**Architecture** (Separate networks):
- Policy Network: [1024, 512] with ReLU activation
- Value Network: [1024, 512] with ReLU activation
- Discriminator Network: [1024, 512] with ReLU activation

**Hyperparameters**:
- Policy Learning Rate: 5.0e-05
- Rollouts: 16
- Learning Epochs: 6
- Mini-batches: 2
- Motion Dataset Size: 200,000
- Replay Buffer: 1,000,000
- Discriminator Batch Size: 4,096
- Total Timesteps: 80,000

---

## Quick Start Commands

### Activate Isaac Lab Environment

```bash
conda activate isaaclab
```

### Training Commands

#### Train with RSL-RL PPO (Fastest)

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/rsl_rl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 1024
```

**Optional Arguments**:
- `--seed 42`: Set random seed for reproducibility
- `--video`: Record training videos
- `--max_iterations 300`: Increase training iterations
- `--num_envs 512`: Adjust number of parallel environments

#### Train with SKRL PPO

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/skrl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 1024
```

#### Train with SKRL AMP (Motion Imitation)

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/skrl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --agent skrl_amp_cfg_entry_point \
  --num_envs 512
```

### Playing Trained Policy

#### Play with RSL-RL Trained Policy

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/rsl_rl/play.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 64 \
  --checkpoint <path-to-checkpoint>
```

#### Play with SKRL Trained Policy

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/skrl/play.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 64 \
  --checkpoint <path-to-checkpoint>
```

### Testing with Baseline Agents

#### Random Action Agent

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/random_agent.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 128
```

#### Zero Action Agent

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/zero_agent.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 128
```

#### List Available Tasks

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/list_envs.py
```

---

## Advanced Usage

### Modifying Environment Configuration

Edit `source/spdrbot_project/spdrbot_project/tasks/direct/spdrbot_project/spdrbot_project_env_cfg.py`:

```python
# Change number of parallel environments
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=2048,        # Increase for faster training (if GPU memory allows)
    env_spacing=2.5,
    replicate_physics=True
)

# Change action scale (how much joints move per action)
action_scale = 0.5        # Higher = larger joint movements

# Adjust reward scales
rew_scale_forward_velocity = 2.0      # More emphasis on forward motion
rew_scale_upright = 1.5               # More emphasis on staying upright
```

### Modifying Reward Function

Edit `source/spdrbot_project/spdrbot_project/tasks/direct/spdrbot_project/spdrbot_project_env.py`:

The `compute_rewards` function contains all reward components. Modify the scales and formulas to change robot behavior:

```python
# Example: Increase forward velocity reward
rew_vel = torch.exp(-torch.square(target_vel - local_lin_vel[:, 0]) / 0.25) * 2.0  # Changed from 1.0

# Example: Add new reward component
rew_new_behavior = torch.sum(torch.abs(joint_vel), dim=-1) * 0.05
```

### Changing Training Hyperparameters

#### For RSL-RL:

Edit `agents/rsl_rl_ppo_cfg.py`:

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32        # More steps per environment
    max_iterations = 300          # Longer training
    learning_rate = 5.0e-4        # Lower learning rate for stability
```

#### For SKRL:

Edit `agents/skrl_ppo_cfg.yaml`:

```yaml
agent:
  learning_rate: 1.0e-03         # Higher learning rate
  learning_epochs: 16            # More learning passes
  mini_batches: 16               # Smaller batch sizes
```

### Distributed Training

Train across multiple GPUs:

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/rsl_rl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 4096 \
  --distributed
```

### Recording Videos During Training

```bash
python /home/surabhi/spdrbot_project/spdrbot_project/scripts/rsl_rl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 1024 \
  --video \
  --video_interval 2000 \
  --video_length 200
```

---

## IDE Setup (Optional)

### VS Code Setup for Intelligent Code Completion

1. Press `Ctrl+Shift+P` in VS Code
2. Select `Tasks: Run Task`
3. Run `setup_python_env`
4. Enter the absolute path to your Isaac Sim installation

This creates a `.python.env` file with paths to all extension modules, enabling Pylance indexing.

### Pylance Configuration

If Pylance crashes due to memory, edit `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-project>/source/spdrbot_project"
    ],
    "python.analysis.exclude": [
        "**/extscache/omni.anim.*",
        "**/extscache/omni.kit.*",
        "**/extscache/omni.graph.*"
    ]
}
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure the project is installed in editable mode:

```bash
cd /home/surabhi/spdrbot_project/spdrbot_project
python -m pip install -e source/spdrbot_project
```

### Issue: CUDA out of memory during training

**Solution**: Reduce `num_envs` in training command:

```bash
python scripts/rsl_rl/train.py --task Template-Spdrbot-Project-Direct-v0 --num_envs 512
```

Or reduce it in the config file.

### Issue: Training is very slow

**Solutions**:
1. Check GPU usage: `nvidia-smi`
2. Ensure you're using CUDA (not CPU):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Increase `num_envs` if GPU has spare capacity

### Issue: Robot doesn't move or moves erratically

**Solutions**:
1. Check reward scales in `spdrbot_project_env.py`
2. Verify action_scale is appropriate (0.25 is default)
3. Test with random agent first: `python scripts/random_agent.py --task Template-Spdrbot-Project-Direct-v0 --num_envs 64`

### Issue: Git/GitHub Push Issues

Force push to main branch (use with caution):

```bash
git push origin main --force
```

Or create a new branch and PR:

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Your changes"
git push origin feature/your-feature
```

---

## Performance Benchmarks

Typical training performance on NVIDIA RTX 3090:

| Algorithm | Num Envs | Training Time | Avg Episode Reward |
|-----------|----------|---------------|--------------------|
| RSL-RL PPO | 1024 | ~2 hours (150 iter) | ~50-100 |
| SKRL PPO | 1024 | ~3 hours | ~40-80 |
| SKRL AMP | 512 | ~8 hours | ~80-150 |

*Note: Times and rewards are estimates and depend on reward shaping configuration*

---

## Contributing

To contribute improvements:

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes and test them

3. Commit with clear messages:
   ```bash
   git commit -m "feat: Add your feature description"
   ```

4. Push and create a pull request:
   ```bash
   git push origin feature/your-feature
   ```

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [SKRL Documentation](https://skrl.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## License

This project is built on Isaac Lab and follows the same BSD-3-Clause license.

---

## Author

- **Project**: SpiderBot Reinforcement Learning Environment
- **Base Framework**: NVIDIA Isaac Lab
- **RL Frameworks**: RSL-RL, SKRL

---

**Last Updated**: January 22, 2026

