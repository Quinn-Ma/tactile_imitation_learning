"""
Evaluate the full 4-stage grasp pipeline in the GraspEnv simulation.

Stage 1  IL policy   → drives arm to object, closes fingers
Stage 2  Micro-lift  → F/T reading before/after 8 mm lift
Stage 3  PINN        → optimal grip force from tactile + Δm

Reported metrics
-----------------
  success_rate   fraction of episodes where object is lifted ≥ 8 cm
  slip_rate      fraction of grip() calls where PINN predicts slip risk
  mean_f_grip    average commanded grip force (N)
  mean_m_real    average estimated object mass (kg)

Quick start
-----------
python scripts/eval_grasp_policy.py \\
    --il_checkpoint  outputs/act_grasp/last \\
    --pinn_checkpoint outputs/pinn_grip/pinn_best.pt \\
    --n_episodes 50 \\
    --object_type random
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_il_policy(checkpoint: str, device: torch.device):
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    except Exception:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy.from_pretrained(checkpoint).to(device).eval()


def _load_pinn(checkpoint: str, device: torch.device):
    from grasp_control.pinn_grip import GripPINN
    model = GripPINN(tactile_dim=12).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _obs_to_batch(obs: dict, device: torch.device) -> dict:
    """Convert numpy observation dict to torch batch (add batch dim)."""
    batch = {}
    for k, v in obs.items():
        t = torch.from_numpy(v.copy()).float()
        if "images" in k:
            # (H, W, 3) → (1, 3, H, W)
            t = t.permute(2, 0, 1).unsqueeze(0)
        else:
            t = t.unsqueeze(0)
        batch[k] = t.to(device)
    return batch


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load models ──────────────────────────────────────────────────────
    logger.info("Loading IL policy from %s …", args.il_checkpoint)
    il_policy = _load_il_policy(args.il_checkpoint, device)

    pinn = None
    if args.pinn_checkpoint:
        logger.info("Loading PINN from %s …", args.pinn_checkpoint)
        pinn = _load_pinn(args.pinn_checkpoint, device)

    from grasp_control import GraspPipeline, GripPINN
    if pinn is None:
        pinn = GripPINN(tactile_dim=12).to(device)
        logger.warning("No PINN checkpoint provided — using random-weight PINN")

    pipeline = GraspPipeline(
        il_policy=il_policy,
        pinn=pinn,
        m_prior=args.m_prior,
        device=device,
    )

    # ── build environment ─────────────────────────────────────────────────
    from grasp_env import GraspEnv
    env = GraspEnv(
        object_type=args.object_type,
        render_mode="human" if args.render else "rgb_array",
    )

    # ── evaluation loop ──────────────────────────────────────────────────
    successes, slips, f_grips, m_reals = [], [], [], []
    ep_returns = []

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=ep)
        pipeline.reset_episode()

        ep_return = 0.0
        contact_detected = False
        ft_baseline = None
        done = False
        step = 0

        while not done and step < args.max_steps:
            # Stage 1: IL policy drives arm to object
            batch = _obs_to_batch(obs, device)
            with torch.no_grad():
                action_t = pipeline.select_arm_action(batch)
            action = action_t.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
            step += 1

            # Detect initial contact: F/T normal force above threshold
            ft = env.get_ft_reading()
            if not contact_detected and np.linalg.norm(ft[:3]) > args.contact_threshold:
                contact_detected = True
                ft_baseline = ft.copy()

                # Stage 2: micro-lift
                env.micro_lift(delta_z=0.008)
                ft_lifted = env.get_ft_reading()
                tactile   = env.get_tactile_reading()

                # Stage 3: PINN grip decision
                is_compliant = (info.get("object") == "foam_ball")
                result = pipeline.grip(
                    ft_baseline=ft_baseline,
                    ft_lifted=ft_lifted,
                    tactile_features=tactile,
                    is_compliant=is_compliant,
                )
                f_grips.append(result.f_grip)
                m_reals.append(result.m_real)
                slips.append(result.slip_risk)

                logger.info(
                    "ep=%3d  object=%-10s  m_real=%.3f kg  Δm=%.3f kg  "
                    "F_grip=%.1f N  slip_risk=%s",
                    ep + 1, info.get("object", "?"),
                    result.m_real, result.delta_m,
                    result.f_grip, result.slip_risk,
                )

                if result.slip_risk:
                    logger.warning("  Slip risk detected — increasing grip to f_min_friction")

        successes.append(done and ep_return > 0)
        ep_returns.append(ep_return)

    # ── summary ─────────────────────────────────────────────────────────
    n = args.n_episodes
    grip_eps = max(len(f_grips), 1)
    logger.info("=" * 55)
    logger.info("Evaluation over %d episodes", n)
    logger.info("  success_rate : %.1f %%  (%d / %d)",
                100 * sum(successes) / n, sum(successes), n)
    logger.info("  slip_rate    : %.1f %%  (%d / %d grip decisions)",
                100 * sum(slips) / grip_eps, sum(slips), grip_eps)
    logger.info("  mean_f_grip  : %.2f N", np.mean(f_grips) if f_grips else float("nan"))
    logger.info("  mean_m_real  : %.4f kg", np.mean(m_reals) if m_reals else float("nan"))
    logger.info("  mean_return  : %.2f", np.mean(ep_returns))
    logger.info("=" * 55)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate full grasp pipeline in GraspEnv")
    parser.add_argument("--il_checkpoint",   required=True,
                        help="Path to trained IL policy checkpoint (LeRobot format)")
    parser.add_argument("--pinn_checkpoint",  default=None,
                        help="Path to trained PINN .pt weights (optional)")
    parser.add_argument("--n_episodes",       type=int,   default=50)
    parser.add_argument("--max_steps",        type=int,   default=500)
    parser.add_argument("--object_type",      choices=["random", "steel", "foam"],
                        default="random")
    parser.add_argument("--m_prior",          type=float, default=0.0,
                        help="Visual mass prior (kg); 0 = no prior")
    parser.add_argument("--contact_threshold",type=float, default=1.0,
                        help="F/T force norm (N) to declare initial contact")
    parser.add_argument("--render",           action="store_true",
                        help="Render the simulation in a window")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
