"""FedSTO: Semi-Supervised Federated Object Detection framework.

Detector-agnostic implementation of the FedSTO algorithm (NeurIPS 2023).
Plug in your detector by subclassing `BaseDetector`.
"""
from .config import FedSTOConfig, OptimConfig
from .detector import BaseDetector, DummyDetector
from .ema import LocalEMA
from .orthogonal import srip_penalty
from .aggregator import fedavg
from .client import Client
from .server import Server
from .orchestrator import FedSTO

__all__ = [
    "FedSTOConfig", "OptimConfig",
    "BaseDetector", "DummyDetector",
    "LocalEMA", "srip_penalty", "fedavg",
    "Client", "Server", "FedSTO",
]
