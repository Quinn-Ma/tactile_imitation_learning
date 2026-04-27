"""
Stage 3: Physics-Informed Neural Network (PINN) grip-force optimizer.

Architecture
------------
Input  : [tactile_features (D,), Δm (1,)]  ← mismatch residual from Stage 2
Output : optimal grip force F_grip (N)

Three physics constraints are encoded as differentiable soft penalties and
added to the supervised MSE loss during training:

  1. Friction-cone constraint
       F_grip ≥ m_eff · g / (2 · μ)
       Prevents object slip; tighter for heavy or low-friction objects.

  2. Actuator force-range constraint
       F_min ≤ F_grip ≤ F_max
       Keeps the command inside safe actuator limits.

  3. Pressure-concentration constraint (compliant objects)
       F_grip ≤ F_soft_max   when is_compliant = 1
       Protects deformable objects from localised over-pressure.

The network is compact enough to run inference in < 0.5 ms on an NVIDIA
Jetson embedded platform (benchmarked at hidden_dim = 64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GRAVITY = 9.81  # m/s²


class GripPINN(nn.Module):
    """
    Compact PINN for online grip-force optimisation.

    Parameters
    ----------
    tactile_dim   : Dimensionality of tactile feature vector.
    hidden_dim    : Width of the two hidden layers.
    mu_friction   : Coefficient of friction μ used in the friction-cone check.
    f_min         : Lower actuator limit (N).
    f_max         : Upper actuator limit (N).
    f_soft_max    : Maximum safe force for compliant / deformable objects (N).
    """

    def __init__(
        self,
        tactile_dim: int = 12,
        hidden_dim: int = 64,
        mu_friction: float = 0.4,
        f_min: float = 1.0,
        f_max: float = 40.0,
        f_soft_max: float = 15.0,
    ) -> None:
        super().__init__()
        self.mu = mu_friction
        self.f_min = f_min
        self.f_max = f_max
        self.f_soft_max = f_soft_max

        # Input: concat(tactile_features, delta_m)
        self.net = nn.Sequential(
            nn.Linear(tactile_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),   # raw ∈ (0, 1), scaled to [f_min, f_max]
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        tactile_features: torch.Tensor,
        delta_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tactile_features : (B, tactile_dim)
        delta_m          : (B,) mismatch residual from micro-lift

        Returns
        -------
        f_grip : (B, 1) predicted grip force in Newtons
        """
        x = torch.cat([tactile_features, delta_m.unsqueeze(-1)], dim=-1)
        raw = self.net(x)                                   # (B, 1) ∈ (0,1)
        return self.f_min + raw * (self.f_max - self.f_min)

    # ------------------------------------------------------------------
    def physics_penalty(
        self,
        f_pred: torch.Tensor,
        m_eff: torch.Tensor,
        is_compliant: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum of constraint violation penalties (all ≥ 0).

        Parameters
        ----------
        f_pred       : (B, 1) predicted grip force
        m_eff        : (B,)   effective object mass estimated from micro-lift
        is_compliant : (B,)   binary flag; 1 = soft/deformable object
        """
        # 1. Friction-cone: prevent slip
        f_min_friction = (m_eff * GRAVITY / (2.0 * self.mu)).unsqueeze(-1)
        slip_penalty = F.relu(f_min_friction - f_pred).mean()

        # 2. Actuator range
        range_penalty = (
            F.relu(self.f_min - f_pred) + F.relu(f_pred - self.f_max)
        ).mean()

        # 3. Pressure concentration for compliant objects
        soft_penalty = (
            is_compliant.unsqueeze(-1) * F.relu(f_pred - self.f_soft_max)
        ).mean()

        return slip_penalty + range_penalty + soft_penalty

    # ------------------------------------------------------------------
    def training_loss(
        self,
        f_pred: torch.Tensor,
        f_target: torch.Tensor,
        m_eff: torch.Tensor,
        is_compliant: torch.Tensor,
        lambda_physics: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Combined data + physics loss.

        Returns a dict with keys 'data', 'physics', 'total' for logging.
        """
        data_loss = F.mse_loss(f_pred, f_target)
        phys_loss = self.physics_penalty(f_pred, m_eff, is_compliant)
        total = data_loss + lambda_physics * phys_loss
        return {"data": data_loss, "physics": phys_loss, "total": total}


# ---------------------------------------------------------------------------
# Thin training helper (no trainer framework dependency)
# ---------------------------------------------------------------------------

def train_pinn(
    model: GripPINN,
    dataset: torch.utils.data.Dataset,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    lambda_physics: float = 1.0,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """
    Train GripPINN on a dataset of (tactile_features, delta_m, f_target,
    m_eff, is_compliant) tuples.

    Returns a list of per-epoch loss dicts for logging / plotting.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        epoch_losses: dict[str, float] = {"data": 0.0, "physics": 0.0, "total": 0.0}
        for batch in loader:
            tf = batch["tactile_features"].to(device)
            dm = batch["delta_m"].to(device)
            ft = batch["f_target"].to(device)
            me = batch["m_eff"].to(device)
            ic = batch["is_compliant"].to(device)

            optimizer.zero_grad()
            f_pred = model(tf, dm)
            loss_dict = model.training_loss(f_pred, ft, me, ic, lambda_physics)
            loss_dict["total"].backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

        n = len(loader)
        history.append({k: v / n for k, v in epoch_losses.items()})

    return history
