"""
Stage 3: Tactile micro-lift mass estimation.

The robot performs an ~8 mm vertical lift immediately after initial finger
contact.  Force-torque (F/T) readings before and after the lift are used to
estimate the object's true mass m_real.  If a visual mass prior m_prior is
available, the mismatch residual Δm = m_real − m_prior is computed and fed
forward to the PINN grip controller as a warning signal.

Sensor convention: force channel index 2 is the vertical axis (z-up).
"""

from __future__ import annotations

import numpy as np


GRAVITY = 9.81          # m/s²
LIFT_HEIGHT_M = 0.008   # 8 mm nominal lift


class TactileMassEstimator:
    """
    Estimate object mass from a F/T sensor pair captured during micro-lift.

    Parameters
    ----------
    ft_noise_threshold : float
        Minimum ΔFz (N) required to trust the reading; filters sensor drift.
    gravity : float
        Gravitational acceleration (m/s²).  Override for non-terrestrial rigs.
    """

    def __init__(
        self,
        ft_noise_threshold: float = 0.05,
        gravity: float = GRAVITY,
    ) -> None:
        self.ft_noise_threshold = ft_noise_threshold
        self.gravity = gravity

    # ------------------------------------------------------------------
    def estimate_mass(
        self,
        ft_baseline: np.ndarray,
        ft_lifted: np.ndarray,
    ) -> float:
        """
        Return estimated object mass (kg) from F/T change along the z-axis.

        Parameters
        ----------
        ft_baseline : (6,) array  [Fx, Fy, Fz, Tx, Ty, Tz] before lift
        ft_lifted   : (6,) array  [Fx, Fy, Fz, Tx, Ty, Tz] after 8 mm lift
        """
        ft_baseline = np.asarray(ft_baseline, dtype=float)
        ft_lifted = np.asarray(ft_lifted, dtype=float)

        if ft_baseline.shape != (6,) or ft_lifted.shape != (6,):
            raise ValueError("F/T arrays must have shape (6,): [Fx,Fy,Fz,Tx,Ty,Tz]")

        delta_fz = ft_lifted[2] - ft_baseline[2]
        if abs(delta_fz) < self.ft_noise_threshold:
            return 0.0
        return abs(delta_fz) / self.gravity

    # ------------------------------------------------------------------
    def compute_mismatch_residual(
        self,
        m_real: float,
        m_prior: float,
    ) -> float:
        """
        Mismatch residual Δm = m_real − m_prior.

        Positive Δm  →  object is heavier than predicted  →  increase grip.
        Negative Δm  →  object is lighter than predicted  →  allow lighter grip.
        Zero m_prior →  no visual estimate available; residual equals m_real.
        """
        return float(m_real - m_prior)

    # ------------------------------------------------------------------
    def run(
        self,
        ft_baseline: np.ndarray,
        ft_lifted: np.ndarray,
        m_prior: float = 0.0,
    ) -> tuple[float, float]:
        """
        Convenience wrapper: estimate mass and compute Δm in one call.

        Returns
        -------
        m_real  : estimated true mass (kg)
        delta_m : mismatch residual Δm (kg)
        """
        m_real = self.estimate_mass(ft_baseline, ft_lifted)
        delta_m = self.compute_mismatch_residual(m_real, m_prior)
        return m_real, delta_m
