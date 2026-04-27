"""
Main grasp pipeline — integrates the three active stages:

  Stage 1  IL policy (ACT or Diffusion)
           Drives the arm to the object and closes the fingers to initial
           contact.  Handled externally; the pipeline receives the already-
           grasped state as inputs.

  Stage 3  Tactile micro-lift (TactileMassEstimator)
           8 mm lift → F/T readings → m_real, Δm

  Stage 4  PINN grip optimiser (GripPINN)
           [tactile_features, Δm] → F_grip

Usage example
-------------
    pipeline = GraspPipeline(pinn=model, m_prior=0.12)
    result = pipeline.execute(
        ft_baseline=ft_before,
        ft_lifted=ft_after,
        tactile_features=sensor_features,
        is_compliant=False,
    )
    send_grip_force(result.f_grip)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from grasp_control.tactile_microlift import TactileMassEstimator
from grasp_control.pinn_grip import GripPINN


@dataclass
class GraspResult:
    """Outcome of one grasp execution."""
    f_grip: float       # commanded grip force (N)
    m_real: float       # estimated object mass (kg)
    delta_m: float      # mismatch residual Δm = m_real − m_prior (kg)
    slip_risk: bool     # True if F_grip is below the friction-cone minimum
    success: bool       # True when slip_risk is False


class GraspPipeline:
    """
    Online grasp controller.

    Parameters
    ----------
    pinn         : Trained GripPINN model (Stage 4).
    m_prior      : Visual mass prior m̂_prior from Stage 2 (kg).
                   Pass 0.0 when Stage 2 is not used.
    ft_noise_thr : F/T noise threshold forwarded to TactileMassEstimator.
    device       : Torch device for PINN inference.
    """

    def __init__(
        self,
        pinn: GripPINN,
        m_prior: float = 0.0,
        ft_noise_thr: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        self.pinn = pinn.to(device).eval()
        self.m_prior = m_prior
        self.device = torch.device(device)
        self._mass_estimator = TactileMassEstimator(ft_noise_threshold=ft_noise_thr)

    # ------------------------------------------------------------------
    def execute(
        self,
        ft_baseline: np.ndarray,
        ft_lifted: np.ndarray,
        tactile_features: np.ndarray,
        is_compliant: bool = False,
    ) -> GraspResult:
        """
        Run Stage 3 → Stage 4 and return a GraspResult.

        Parameters
        ----------
        ft_baseline      : (6,) F/T reading before micro-lift [Fx…Tz]
        ft_lifted        : (6,) F/T reading after ~8 mm lift
        tactile_features : (D,) processed tactile sensor features
        is_compliant     : True if the object is soft / deformable
        """
        # ---- Stage 3: micro-lift ----------------------------------------
        m_real, delta_m = self._mass_estimator.run(ft_baseline, ft_lifted, self.m_prior)

        # ---- Stage 4: PINN grip optimisation ----------------------------
        with torch.no_grad():
            tf = torch.tensor(tactile_features, dtype=torch.float32, device=self.device)
            dm = torch.tensor([delta_m], dtype=torch.float32, device=self.device)
            f_grip_t = self.pinn(tf.unsqueeze(0), dm)   # (1, 1)

        f_grip = float(f_grip_t.squeeze())

        # ---- Slip-risk check (informational) ----------------------------
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
        """Update the visual mass prior at runtime (e.g., between objects)."""
        self.m_prior = float(m_prior)
