# Vision-Tactile Grasping with Online Physics-Informed Grip Optimization

This repository contains the implementation of a modular robot grasping framework that combines imitation learning with tactile feedback and physics-informed grip force control. The system enables robust grasping of objects with unknown physical properties (mass, compliance) by fusing visual approach, tactile sensing, and online force optimization.

## Method Overview

The pipeline consists of three active stages executed sequentially at grasp time:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1  Imitation Learning (IL) Policy                        │
│           ACT or Diffusion Policy (via LeRobot)                 │
│           → drives arm to object, closes fingers to contact     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3  Tactile Micro-Lift                                    │
│           8 mm vertical lift + F/T sensor reading               │
│           → estimates true mass m_real, computes Δm residual    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4  PINN Grip Optimizer                                   │
│           Physics-Informed Neural Network                       │
│           → optimal grip force F_grip with 3 hard constraints:  │
│              · Friction cone   (prevents slip)                  │
│              · Actuator range  (safe motor limits)              │
│              · Pressure limit  (protects compliant objects)     │
└─────────────────────────────────────────────────────────────────┘
```

**Key results** (steel ball vs. soft ball, visually indistinguishable):
- Grasp success rate: **86.3%**
- Peak grip force reduction: **27–75 N** vs. baselines
- Inference latency on NVIDIA Jetson: **< 0.5 ms**

## Repository Structure

```
tactile_imitation_learning/
├── grasp_control/                  # Core paper contributions
│   ├── tactile_microlift.py        # Stage 3: F/T-based mass estimation
│   ├── pinn_grip.py                # Stage 4: PINN grip force optimizer
│   └── grasp_pipeline.py          # Integrated 3-stage pipeline
│
├── grasp_env/
│   ├── __init__.py                 # Gymnasium registration (GraspEnv-v0)
│   └── grasp_env.py               # MuJoCo simulation environment
│
├── scripts/
│   ├── collect_grasping_demos.py  # Record demonstrations → LeRobot dataset
│   ├── train_il_policy.py         # Train ACT or Diffusion Policy (Stage 1)
│   ├── train_pinn.py              # Train PINN grip optimizer (Stage 4)
│   └── eval_grasp_policy.py       # Full pipeline evaluation in simulation
│
├── configs/
│   ├── act_grasp.yaml             # ACT hyperparameters for grasping
│   └── diffusion_grasp.yaml       # Diffusion Policy hyperparameters
│
├── irl_control/                   # MuJoCo OSC robot controller
│   ├── robot.py                   # Robot state management
│   ├── osc.py                     # Operational space controller
│   └── assets/
│       ├── grasp_scene.xml        # Single-arm grasping scene
│       └── meshes/                # UR5 + Robotiq 85 STL meshes
│
└── requirements/
    └── requirements.txt
```

## Installation

```bash
git clone https://github.com/Quinn-Ma/tactile_imitation_learning.git
cd tactile_imitation_learning
pip install -e .
pip install lerobot>=0.4.0
```

**Dependencies** (installed automatically via `setup.py`):
- [LeRobot](https://github.com/huggingface/lerobot) ≥ 0.4.0 — IL policies and dataset management
- PyTorch ≥ 2.0
- MuJoCo ≥ 3.0
- Gymnasium ≥ 0.28

## Usage

### Step 1 — Collect Demonstrations

Record teleoperated grasping demonstrations and store them as a LeRobot dataset.

```bash
python scripts/collect_grasping_demos.py \
    --repo_id myorg/grasp_demos \
    --root   ./data/grasp_demos \
    --n_episodes 50
```

The dataset stores per-frame observations as a **25-dim state vector**:

| Indices | Content |
|---------|---------|
| `[0:6]` | Arm joint positions (UR5, 6-DOF) |
| `[6]` | Gripper aperture ∈ [0, 1] |
| `[7:13]` | Wrist F/T sensor `[Fx Fy Fz Tx Ty Tz]` |
| `[13:25]` | Tactile array (12 values, 6 per fingerpad) |

Connect your hardware by implementing `RobotInterface` and `TeleopController` in the script.

### Step 2 — Train the IL Policy (Stage 1)

Train an ACT or Diffusion Policy on the collected demonstrations:

```bash
# ACT (recommended for precise, contact-rich tasks)
python scripts/train_il_policy.py \
    --policy act \
    --repo_id myorg/grasp_demos \
    --output_dir outputs/act_grasp \
    --steps 80000

# Diffusion Policy
python scripts/train_il_policy.py \
    --policy diffusion \
    --repo_id myorg/grasp_demos \
    --output_dir outputs/diffusion_grasp \
    --steps 80000
```

Hyperparameters are in [`configs/act_grasp.yaml`](configs/act_grasp.yaml) and [`configs/diffusion_grasp.yaml`](configs/diffusion_grasp.yaml).

### Step 3 — Train the PINN Grip Optimizer (Stage 4)

Prepare a `.npz` dataset with fields `tactile_features`, `delta_m`, `f_target`, `m_eff`, `is_compliant` (see `GripDataset` in `scripts/train_pinn.py`), then:

```bash
python scripts/train_pinn.py \
    --data_path  data/pinn_dataset.npz \
    --output_dir outputs/pinn_grip \
    --epochs 200 \
    --lambda_physics 1.0
```

Or train programmatically:

```python
from grasp_control.pinn_grip import GripPINN, train_pinn

model = GripPINN(
    tactile_dim=12,
    hidden_dim=64,
    mu_friction=0.4,
    f_min=1.0,
    f_max=40.0,
    f_soft_max=15.0,
)
history = train_pinn(model, dataset, epochs=200, lambda_physics=1.0)
```

The PINN loss combines supervised MSE with three physics penalty terms:

```
L = L_data + λ · (L_friction + L_range + L_pressure)
```

### Step 4 — Evaluate in Simulation

Run the full pipeline in `GraspEnv` (MuJoCo):

```bash
python scripts/eval_grasp_policy.py \
    --il_checkpoint  outputs/act_grasp/last \
    --pinn_checkpoint outputs/pinn_grip/pinn_best.pt \
    --n_episodes 50 \
    --object_type random
```

Or use the pipeline API directly:

```python
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from grasp_control import GraspPipeline, GripPINN
from grasp_env import GraspEnv

# Load trained models
il_policy = ACTPolicy.from_pretrained("outputs/act_grasp/last")
pinn = GripPINN(tactile_dim=12)
pinn.load_state_dict(torch.load("outputs/pinn_grip/pinn_best.pt"))

pipeline = GraspPipeline(il_policy=il_policy, pinn=pinn, m_prior=0.0)
env = GraspEnv(object_type="random", render_mode="human")

obs, _ = env.reset()
pipeline.reset_episode()

# Stage 1: arm approach (call in control loop)
import numpy as np
done = False
while not done:
    batch = {k: torch.from_numpy(v).float().unsqueeze(0)
             for k, v in obs.items() if "state" in k}
    batch["observation.images.wrist"] = (
        torch.from_numpy(obs["observation.images.wrist"])
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    )
    action = pipeline.select_arm_action(batch).squeeze(0).numpy()
    obs, reward, done, _, info = env.step(action)

    # Detect contact → Stage 3 + 4
    ft = env.get_ft_reading()
    if np.linalg.norm(ft[:3]) > 1.0:
        ft_baseline = ft.copy()
        env.micro_lift(delta_z=0.008)
        result = pipeline.grip(
            ft_baseline=ft_baseline,
            ft_lifted=env.get_ft_reading(),
            tactile_features=env.get_tactile_reading(),
            is_compliant=(info["object"] == "foam_ball"),
        )
        print(f"F_grip={result.f_grip:.1f} N  m_real={result.m_real:.3f} kg  "
              f"slip_risk={result.slip_risk}")
        break
```

## Module Reference

### `grasp_env.GraspEnv`

MuJoCo Gymnasium environment for single-arm grasping.

| Property | Value |
|----------|-------|
| Observation | Dict: `observation.state` (25,) + `observation.images.wrist` (120×160×3) |
| Action | (7,): 6 joint position targets + 1 gripper aperture ∈ [0, 1] |
| Reward | +10 lift success, −0.1/step, −5 drop |
| Scene | UR5 + Robotiq 85, table, steel ball, foam ball |

```python
env = GraspEnv(object_type="random")  # or "steel" / "foam"
ft  = env.get_ft_reading()            # (6,) [Fx Fy Fz Tx Ty Tz]
tac = env.get_tactile_reading()       # (12,) fingerpad contact forces
env.micro_lift(delta_z=0.008)         # trigger Stage 3 micro-lift
```

### `grasp_control.tactile_microlift.TactileMassEstimator`

Estimates object mass from a pair of F/T readings bracketing the micro-lift.

```python
estimator = TactileMassEstimator(ft_noise_threshold=0.05)

m_real = estimator.estimate_mass(ft_baseline, ft_lifted)        # kg
delta_m = estimator.compute_mismatch_residual(m_real, m_prior)  # kg

# Or in one call:
m_real, delta_m = estimator.run(ft_baseline, ft_lifted, m_prior)
```

### `grasp_control.pinn_grip.GripPINN`

Compact MLP with physics-constraint penalties. Input: `[tactile_features, Δm]`. Output: grip force (N).

```python
model = GripPINN(tactile_dim=12, hidden_dim=64, mu_friction=0.4,
                 f_min=1.0, f_max=40.0, f_soft_max=15.0)

f_grip = model(tactile_features, delta_m)        # inference
loss   = model.training_loss(f_pred, f_target,   # training
                              m_eff, is_compliant,
                              lambda_physics=1.0)
```

### `grasp_control.grasp_pipeline.GraspPipeline`

Integrates all active stages. Wraps a LeRobot `PreTrainedPolicy` for Stage 1.

| Method | Description |
|--------|-------------|
| `reset_episode()` | Calls `il_policy.reset()` before each grasp |
| `select_arm_action(obs_batch)` | Stage 1 inference via `il_policy.select_action()` |
| `grip(ft_baseline, ft_lifted, tactile_features)` | Stage 3 + 4: returns `GraspResult` |
| `update_prior(m_prior)` | Update visual mass prior at runtime |

## Citation

```bibtex
@article{ma2025tactile,
  title   = {Vision-Tactile Mass Priors with Online Physics-Informed Grip Optimization},
  author  = {Ma, Quinn},
  year    = {2025},
}
```

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face) for ACT and Diffusion Policy implementations and the LeRobot dataset format.
- [ACT](https://github.com/tonyzhaozh/act) — Tony Z. Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.
- [Diffusion Policy](https://github.com/columbia-ai-robotics/diffusion_policy) — Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*.
