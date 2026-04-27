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
├── scripts/
│   ├── collect_grasping_demos.py  # Record demonstrations → LeRobot dataset
│   └── train_il_policy.py         # Train ACT or Diffusion Policy (Stage 1)
│
├── configs/
│   ├── act_grasp.yaml             # ACT hyperparameters for grasping
│   └── diffusion_grasp.yaml       # Diffusion Policy hyperparameters
│
├── irl_control/                   # MuJoCo OSC robot controller
│   ├── robot.py                   # Robot state management
│   ├── osc.py                     # Operational space controller
│   └── assets/                    # Robot URDF/XML meshes (UR5 + Robotiq)
│
└── requirements/
    └── requirements.txt
```

## Installation

```bash
git clone https://github.com/Quinn-Ma/tactile_imitation_learning.git
cd tactile_imitation_learning
pip install -e .
```

**Dependencies** (installed automatically):
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

The dataset stores per-frame observations as a 26-dim state vector:

| Indices | Content |
|---------|---------|
| `[0:7]` | Arm joint positions |
| `[7]` | Gripper aperture |
| `[8:14]` | Wrist F/T sensor `[Fx Fy Fz Tx Ty Tz]` |
| `[14:26]` | Tactile array (12 taxels) |

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

### Step 4 — Run the Full Pipeline

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from grasp_control import GraspPipeline, GripPINN

# Load trained models
il_policy = ACTPolicy.from_pretrained("outputs/act_grasp/last")
pinn      = GripPINN(tactile_dim=12)
# ... load pinn weights ...

# Build pipeline
pipeline = GraspPipeline(il_policy=il_policy, pinn=pinn, m_prior=0.0)

# At each episode
pipeline.reset_episode()

# Stage 1: arm approach (call in control loop)
action = pipeline.select_arm_action({
    "observation.state":         obs_state_tensor,   # (1, 26)
    "observation.images.wrist":  wrist_image_tensor, # (1, 3, H, W)
})
robot.send_action(action)

# Stage 3 + 4: after initial contact, perform micro-lift and optimize grip
result = pipeline.grip(
    ft_baseline=ft_before_lift,   # (6,) F/T before 8 mm lift
    ft_lifted=ft_after_lift,      # (6,) F/T after lift
    tactile_features=taxel_array, # (12,) tactile readings
    is_compliant=False,
)
print(f"F_grip={result.f_grip:.1f} N  m_real={result.m_real:.3f} kg  slip_risk={result.slip_risk}")
robot.set_grip_force(result.f_grip)
```

## Module Reference

### `grasp_control.tactile_microlift.TactileMassEstimator`

Estimates object mass from a pair of F/T readings bracketing the micro-lift.

```python
estimator = TactileMassEstimator(ft_noise_threshold=0.05)

m_real = estimator.estimate_mass(ft_baseline, ft_lifted)   # kg
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
