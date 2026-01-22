# SpiderBot – Quadruped Locomotion in Isaac Lab

This repository contains my custom reinforcement learning environment for training a **12-joint quadruped "spiderbot"** to walk using **NVIDIA Isaac Lab**.

The core contributions are:

* A custom robot asset loaded from USD
* A locomotion RL environment
* A reward function for stable walking
* PPO training configs (RSL-RL + SKRL)
* Gym registration so the task runs inside Isaac Lab

---

## What I Built

* Imported a CAD → URDF → USD quadruped into Isaac Sim
* Wrapped the robot as an Isaac Lab **Articulation**
* Designed a custom **DirectRLEnv**
* Defined observations, actions, rewards, and termination
* Trained using **PPO** (RSL-RL & SKRL)

This is not a template — all logic is custom for this robot.

---

## Quick Setup

### 1. Activate Isaac Lab

```bash
conda activate isaaclab
```

### 2. Clone & Install

```bash
git clone https://github.com/surabhigade/spdrbot_project.git
cd spdrbot_project
python -m pip install -e source/spdrbot_project
```

### 3. Set Robot USD Path

Edit:

```
source/isaaclab_assets/robots/spdrbot.py
```

Update:

```python
usd_path="/path/to/your/spdrbot_video.usd"
```

---

## Run Checks

List environments:

```bash
python scripts/list_envs.py
```

Test with random actions:

```bash
python scripts/random_agent.py --task Template-Spdrbot-Project-Direct-v0 --num_envs 64
```

---

## Train

### PPO (RSL-RL)

```bash
python scripts/rsl_rl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 1024
```

### PPO (SKRL)

```bash
python scripts/skrl/train.py \
  --task Template-Spdrbot-Project-Direct-v0 \
  --num_envs 1024
```

---

## Key Files I Modified

```
source/
├── isaaclab_assets/robots/spdrbot.py      # Robot asset (USD + actuators)
└── spdrbot_project/tasks/direct/spdrbot_project/
    ├── spdrbot_project_env.py             # RL environment logic
    ├── spdrbot_project_env_cfg.py         # Training configuration
    ├── __init__.py                        # Gym registration
    └── agents/                            # PPO configs
```

---

## Environment Summary

* **Robot**: 12 revolute joints
* **Actions**: 12 joint targets
* **Observations**: 45D (velocity, gravity, joints, actions)
* **Control rate**: 30 Hz
* **Parallel envs**: 1024 (GPU)

---

## Notes

* Training is GPU-only (CPU is very slow)
* Reward tuning is inside `compute_rewards()`
* Gait emerges from reward + physics (not scripted)

---

**Author**: Surabhi Gade
**Framework**: NVIDIA Isaac Lab
#find /home/surabhi/spdrbot_project -name "__pycache__" -type d -exec rm -rf {} +

#python /home/surabhi/spdrbot_project/spdrbot_project/scripts/rsl_rl/train.py --task Template-Spdrbot-Project-Direct-v0  --num_envs 1024

#source /home/surabhi/isaacsim/setup_conda_env.sh

#-ubuntu:~/isaacsim$ ------  ./isaac-sim.selector.sh
