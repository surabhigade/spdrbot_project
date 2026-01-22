# SpiderBot – Quadruped Locomotion in Isaac Lab

This repository contains my custom reinforcement learning environment for training a **12-joint quadruped spiderbot** to walk using **NVIDIA Isaac Lab**.

The core contributions are:

* A custom robot asset loaded from USD
* A locomotion RL environment
* A reward function for stable walking
* PPO and AMP training configs (RSL-RL + SKRL)
* Gym registration so the task runs inside Isaac Lab

---
Refer blog for detailed description: https://medium.com/p/9e5818686f05

---
## What I Built

* Imported a CAD → URDF → USD quadruped into Isaac Sim
* Wrapped the robot as an Isaac Lab **Articulation**
* Designed a custom **DirectRLEnv**
* Defined observations, actions, rewards, and termination
* Trained using **PPO** and **AMP**(RSL-RL & SKRL)

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

