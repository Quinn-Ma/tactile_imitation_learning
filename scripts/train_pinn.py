"""
Train the PINN grip-force optimizer (Stage 3) on recorded grasp data.

Input data format (torch Dataset)
-----------------------------------
Each sample is a dict with keys:
    tactile_features  float32 (tactile_dim,)  from fingerpad contacts
    delta_m           float32 scalar           m_real − m_prior
    f_target          float32 scalar           expert grip force (N)
    m_eff             float32 scalar           estimated object mass (kg)
    is_compliant      float32 scalar           1.0 = soft object

Quick start
-----------
python scripts/train_pinn.py \\
    --data_path data/pinn_dataset.npz \\
    --output_dir outputs/pinn_grip \\
    --epochs 200 \\
    --lambda_physics 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from grasp_control.pinn_grip import GripPINN

logger = logging.getLogger(__name__)


class GripDataset(Dataset):
    """
    Dataset loaded from a .npz file with arrays:
        tactile_features : (N, tactile_dim)
        delta_m          : (N,)
        f_target         : (N,)
        m_eff            : (N,)
        is_compliant     : (N,)
    """

    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(npz_path)
        self.tf = torch.tensor(data["tactile_features"], dtype=torch.float32)
        self.dm = torch.tensor(data["delta_m"],          dtype=torch.float32)
        self.ft = torch.tensor(data["f_target"],         dtype=torch.float32).unsqueeze(-1)
        self.me = torch.tensor(data["m_eff"],            dtype=torch.float32)
        self.ic = torch.tensor(data["is_compliant"],     dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tf)

    def __getitem__(self, idx):
        return {
            "tactile_features": self.tf[idx],
            "delta_m":          self.dm[idx],
            "f_target":         self.ft[idx],
            "m_eff":            self.me[idx],
            "is_compliant":     self.ic[idx],
        }


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training PINN on %s", device)

    dataset = GripDataset(args.data_path)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    tactile_dim = dataset.tf.shape[-1]
    model = GripPINN(
        tactile_dim=tactile_dim,
        hidden_dim=args.hidden_dim,
        mu_friction=args.mu_friction,
        f_min=args.f_min,
        f_max=args.f_max,
        f_soft_max=args.f_soft_max,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: dict[str, float] = {"data": 0.0, "physics": 0.0, "total": 0.0}
        for batch in train_loader:
            tf = batch["tactile_features"].to(device)
            dm = batch["delta_m"].to(device)
            ft = batch["f_target"].to(device)
            me = batch["m_eff"].to(device)
            ic = batch["is_compliant"].to(device)

            optimizer.zero_grad()
            f_pred = model(tf, dm)
            loss_d = model.training_loss(f_pred, ft, me, ic, args.lambda_physics)
            loss_d["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += loss_d[k].item()

        n = len(train_loader)
        train_losses = {k: v / n for k, v in train_losses.items()}

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tf = batch["tactile_features"].to(device)
                dm = batch["delta_m"].to(device)
                ft = batch["f_target"].to(device)
                me = batch["m_eff"].to(device)
                ic = batch["is_compliant"].to(device)
                f_pred = model(tf, dm)
                ld = model.training_loss(f_pred, ft, me, ic, args.lambda_physics)
                val_total += ld["total"].item()
        val_total /= max(len(val_loader), 1)

        if epoch % args.log_freq == 0:
            logger.info(
                "epoch=%4d  train=%.4f (data=%.4f phys=%.4f)  val=%.4f",
                epoch, train_losses["total"],
                train_losses["data"], train_losses["physics"], val_total,
            )

        if val_total < best_val:
            best_val = val_total
            ckpt_path = output_dir / "pinn_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  → new best checkpoint (val=%.4f) → %s", best_val, ckpt_path)

    # Save final weights
    final_path = output_dir / "pinn_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Training complete. Final → %s  Best → %s/pinn_best.pt",
                final_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train GripPINN (Stage 3)")
    parser.add_argument("--data_path",    required=True, help=".npz file with grip dataset")
    parser.add_argument("--output_dir",   default="outputs/pinn_grip")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden_dim",   type=int,   default=64)
    parser.add_argument("--lambda_physics", type=float, default=1.0)
    parser.add_argument("--mu_friction",  type=float, default=0.4)
    parser.add_argument("--f_min",        type=float, default=1.0)
    parser.add_argument("--f_max",        type=float, default=40.0)
    parser.add_argument("--f_soft_max",   type=float, default=15.0)
    parser.add_argument("--log_freq",     type=int,   default=20)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
