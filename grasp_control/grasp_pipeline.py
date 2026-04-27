"""
Main grasp pipeline — integrates Stage 1 (LeRobot IL policy), Stage 3
(tactile micro-lift), and Stage 4 (PINN grip force optimiser).

Stage 1  IL policy  (lerobot.policies.act.ACTPolicy  OR
                     lerobot.policies.diffusion.DiffusionPolicy)
         Drives the arm to the object and closes the fingers to initial
         contact.  The policy is called via the standard LeRobot API:
             policy.reset()
             action = policy.select_action(batch)

Stage 3  TactileMassEstimator
         8 mm vertical micro-lift → Δ F/T → m_real, Δm

Stage 4  GripPINN
         [tactile_features, Δm] → optimal grip force F_grip

Usage
-----
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig

    il_policy = ACTPolicy.from_pretrained("path/to/checkpoint")
    pipeline  = GraspPipeline(il_policy=il_policy, pinn=grip_pinn)

    il_policy.reset()
    # ... arm moves to object (Stage 1 drives joint actions) ...
    result = pipeline.grip(
        ft_baseline=ft_before,
        ft_lifted=ft_after,
        tactile_features=sensor_features,
    )
    send_gripper_command(result.f_grip)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from grasp_control.tactile_microlift import TactileMassEstimator
from grasp_control.pinn_grip import GripPINN


@dataclass
class GraspResult:
    """Outcome of one grip-force decision."""
    f_grip: float     # commanded grip force (N)
    m_real: float     # estimated object mass from micro-lift (kg)
    delta_m: float    # mismatch residual Δm = m_real − m_prior (kg)
    slip_risk: bool   # True when F_grip < friction-cone minimum
    success: bool     # not slip_risk


class GraspPipeline:
    """
    Online grasp controller integrating Stage 1 (LeRobot IL policy) with
    the tactile micro-lift (Stage 3) and PINN grip optimiser (Stage 4).

    Parameters
    ----------
    il_policy    : A LeRobot PreTrainedPolicy (ACTPolicy or DiffusionPolicy).
                   Must implement ``reset()`` and ``select_action(batch)``.
    pinn         : Trained GripPINN model.
    m_prior      : Visual mass prior (kg).  Pass 0.0 when Stage 2 is absent.
    ft_noise_thr : F/T noise floor forwarded to TactileMassEstimator.
    device       : Torch device for PINN inference.
    """

    def __init__(
        self,
        il_policy,
        pinn: GripPINN,
        m_prior: float = 0.0,
        ft_noise_thr: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        self.il_policy = il_policy
        self.pinn = pinn.to(device).eval()
        self.m_prior = m_prior
        self.device = torch.device(device)
        self._mass_estimator = TactileMassEstimator(ft_noise_threshold=ft_noise_thr)

    # ------------------------------------------------------------------
    # Stage 1 helpers — thin wrappers around the LeRobot policy API
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Call before every new grasp episode to clear policy state."""
        self.il_policy.reset()

    def select_arm_action(self, obs_batch: dict[str, Tensor]) -> Tensor:
        """
        Stage 1 inference: select the next arm joint action.

        Parameters
        ----------
        obs_batch : dict following the LeRobot observation convention, e.g.
            {
                "observation.state":        Tensor (1, state_dim),
                "observation.images.wrist": Tensor (1, C, H, W),
            }

        Returns
        -------
        action : Tensor (action_dim,)
        """
        return self.il_policy.select_action(obs_batch)

    # ------------------------------------------------------------------
    # Stage 3 + 4 — grip force decision after initial contact
    # ------------------------------------------------------------------

    def grip(
        self,
        ft_baseline: np.ndarray,
        ft_lifted: np.ndarray,
        tactile_features: np.ndarray,
        is_compliant: bool = False,
    ) -> GraspResult:
        """
        Run Stage 3 (micro-lift) → Stage 4 (PINN) and return grip decision.

        Parameters
        ----------
        ft_baseline      : (6,) F/T reading before micro-lift [Fx Fy Fz Tx Ty Tz]
        ft_lifted        : (6,) F/T reading after ~8 mm lift
        tactile_features : (D,) processed tactile sensor features
        is_compliant     : True if object is soft / deformable
        """
        # Stage 3
        m_real, delta_m = self._mass_estimator.run(ft_baseline, ft_lifted, self.m_prior)

        # Stage 4
        with torch.no_grad():
            tf = torch.tensor(tactile_features, dtype=torch.float32, device=self.device)
            dm = torch.tensor([delta_m], dtype=torch.float32, device=self.device)
            f_grip_t = self.pinn(tf.unsqueeze(0), dm)

        f_grip = float(f_grip_t.squeeze())
        f_min_friction = m_real * 9.81 / (2.0 * self.pinn.mu)
        slip_risk = f_grip < f_min_friction

        return GraspResult(
            f_grip=f_grip,
            m_real=m_real,
            delta_m=delta_m,
            slip_risk=slip_risk,
            success=not slip_risk,
        )

    # ------------------------------------------------------------------

    def update_prior(self, m_prior: float) -> None:
        """Update visual mass prior at runtime (between objects)."""
        self.m_prior = float(m_prior)
