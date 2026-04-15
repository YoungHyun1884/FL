"""FedSTO configuration.

All knobs in one place. Sizes/steps default to small values so the synthetic
main.py runs in seconds; scale up for real datasets.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class OptimConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class FedSTOConfig:
    # Rounds (paper: 300 total = 50 warmup + 100 T1 + 150 T2)
    warmup_rounds: int = 50           # server supervised pretraining rounds
    T1: int = 100                     # Phase 1 (Selective Training) rounds
    T2: int = 150                     # Phase 2 (FPT + Orthogonal) rounds
    local_epochs: int = 1             # local epochs per client per round (paper: 1)
    server_steps: int = 20            # server supervised steps after aggregation

    # Federated sampling
    num_clients: int = 3              # paper: 3 (Clear / Night / Adverse)
    client_sample_ratio: float = 1.0  # 1.0 = all clients every round

    # Local EMA teacher
    ema_decay: float = 0.999
    reset_ema_each_round: bool = True # re-init local EMA from broadcast global

    # Pseudo-label thresholds (paper: ignore threshold 0.1 ~ 0.6)
    tau_low: float = 0.1              # NMS confidence threshold / soft boundary
    tau_high: float = 0.6             # hard pseudo-label boundary

    # Epoch Adaptor: linearly schedule thresholds over training
    use_epoch_adaptor: bool = True
    tau_low_start: float = 0.1        # tau_low at training start (constant per paper)
    tau_high_start: float = 0.3       # tau_high at training start

    # Unsupervised loss weight
    unsup_loss_weight: float = 4.0

    # Orthogonal regularization (SRIP)
    ortho_lambda: float = 1e-3
    ortho_power_iters: int = 1        # power iteration steps for spectral norm

    # Optimizers
    server_opt: OptimConfig = field(default_factory=OptimConfig)
    client_opt: OptimConfig = field(default_factory=OptimConfig)

    # Runtime
    device: str = "cuda"
    seed: int = 42

    # Logging
    log_every: int = 1
    ckpt_dir: str = "./checkpoints"

    def get_thresholds(self, progress: float) -> Tuple[float, float]:
        """Return (tau_low, tau_high) for the given training progress in [0, 1].

        When ``use_epoch_adaptor`` is True the thresholds are linearly
        interpolated from their ``*_start`` values to the final values,
        mimicking the Efficient Teacher Epoch Adaptor.
        """
        progress = max(0.0, min(1.0, progress))
        if not self.use_epoch_adaptor:
            return self.tau_low, self.tau_high
        tl = self.tau_low_start + (self.tau_low - self.tau_low_start) * progress
        th = self.tau_high_start + (self.tau_high - self.tau_high_start) * progress
        return tl, th
