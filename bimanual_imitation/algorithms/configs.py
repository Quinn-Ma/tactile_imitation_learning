from dataclasses import dataclass
from enum import Enum


class ALG(Enum):
    BCLONE = "bclone"
    EXPERT = "expert"
    DIFFUSION = "diffusion"
    ACT = "act"


@dataclass(frozen=True)
class BcloneParamConfig:
    bclone_lr: float = 5e-4
    obsnorm_mode: str = "expertdata"
    policy_n_layers: int = 3
    policy_n_units: int = 512
    policy_layer_type: str = "tanh"
    continuous_policy_type: str = "Gaussian"
    deterministic_eval: bool = False
    bclone_l1_lambda: float = 1e-4
    bclone_l2_lambda: float = 1e-4
    bclone_batch_size: int = 128


@dataclass(frozen=True)
class DiffusionParamConfig:
    batch_size: int = 512
    num_diffusion_iters: int = 100
    opt_learning_rate: float = 1.0e-4
    opt_weight_decay: float = 1.0e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    obs_horizon: int = 4
    action_horizon: int = 4
    pred_horizon: int = 8


@dataclass(frozen=True)
class ActParamConfig:
    batch_size: int = 256
    enc_layers: int = 1
    dec_layers: int = 1
    latent_dim: int = 8
    n_heads: int = 4
    act_lr: float = 1.0e-4
    dropout: float = 0.1
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    dim_feedforward: int = 512
    activation: str = "relu"
    kl_weight: int = 100
    pre_norm: bool = False
    obs_horizon: int = 4
    action_horizon: int = 4
    pred_horizon: int = 8


def get_param_config(alg: ALG):
    param_cfgs = {
        ALG.BCLONE: BcloneParamConfig,
        ALG.DIFFUSION: DiffusionParamConfig,
        ALG.ACT: ActParamConfig,
    }
    return param_cfgs[alg]
