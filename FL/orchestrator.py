"""논문 알고리즘 1 전체를 실제 코드 흐름으로 돌리는 파일
"""
from __future__ import annotations
import os
import random
from dataclasses import asdict
from typing import Callable, List

import torch

from .aggregator import fedavg
from .client import Client
from .config import FedSTOConfig
from .detector import BaseDetector
from .server import Server


class FedSTO:
    def __init__(
        self,
        global_model: BaseDetector,
        server: Server,
        clients: List[Client],
        cfg: FedSTOConfig,
        eval_fn: Callable[[BaseDetector, int, str], dict] | None = None,
    ):
        self.global_model = global_model.to(cfg.device)
        self.server = server
        self.clients = clients
        self.cfg = cfg
        self.eval_fn = eval_fn
        self.history: list[dict] = []
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # ------- 보조 함수 -----------------------------------------------------
    def _sample_clients(self) -> List[Client]:
        k = max(1, int(round(self.cfg.client_sample_ratio * len(self.clients))))
        return random.sample(self.clients, k)

    def _broadcast_global(self) -> None:
        # 서버가 레이블 데이터로 보정한 최신 모델을 전역 기준 모델로 맞춘다.
        self.global_model.load_state_dict(self.server.model.state_dict())

    def _pull_into_server(self, sd: dict) -> None:
        self.server.model.load_state_dict(sd, strict=False)

    def _log(self, phase: str, rnd: int, **kv) -> None:
        rec = {"phase": phase, "round": rnd, **kv}
        self.history.append(rec)
        if rnd % self.cfg.log_every == 0:
            kv_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in kv.items())
            print(f"[{phase}][r{rnd:03d}] {kv_str}")

    def _save_ckpt(self, tag: str) -> None:
        path = os.path.join(self.cfg.ckpt_dir, f"global_{tag}.pt")
        torch.save(
            {
                "state_dict": self.global_model.state_dict(),
                "config": asdict(self.cfg),
                "history": self.history,
            },
            path,
        )
        print(f"  -> saved {path}")

    # ------- 메인 실행 흐름 ----------------------------------------------------
    def run(self) -> None:
        # ---------- 사전학습 ----------
        # 0단계) 서버만 레이블 데이터로 먼저 학습한다.
        print("=== Warmup (server supervised) ===")
        warm_loss = self.server.warmup()
        self._log("warmup", 0, loss=warm_loss)
        self._broadcast_global()
        self._save_ckpt("warmup")
        if self.eval_fn:
            self.eval_fn(self.global_model, 0, "warmup")

        # ---------- 1단계: 선택적 학습 ----------
        print("=== Phase 1: Selective Training ===")
        total_rounds = self.cfg.T1 + self.cfg.T2
        for rnd in range(self.cfg.T1):
            # Epoch Adaptor는 라운드가 진행될수록 가짜 라벨 기준을 점점 엄격하게 만든다.
            progress = rnd / max(total_rounds - 1, 1)
            tau_low, tau_high = self.cfg.get_thresholds(progress)

            # 1단계) 일부 또는 전체 클라이언트를 선택해 현재 전역 모델 기준으로 로컬 학습한다.
            selected = self._sample_clients()
            client_updates = []
            for c in selected:
                upd = c.train_phase1(self.global_model, tau_low=tau_low, tau_high=tau_high)
                client_updates.append(upd)

            # 2단계) 1단계에서는 backbone만 FedAvg로 합친다.
            agg_backbone = fedavg(
                [u["backbone_state_dict"] for u in client_updates],
                [u["num_samples"] for u in client_updates],
            )
            agg_backbone = {k: v.to(self.cfg.device) for k, v in agg_backbone.items()}
            self.global_model.load_backbone_state_dict(agg_backbone)

            # 3단계) 집계된 전역 모델을 서버로 가져와 레이블 데이터로 보정한다.
            self.server.model.load_state_dict(self.global_model.state_dict())
            server_loss = self.server.update(use_ortho=False)
            # 4단계) 보정이 끝난 서버 모델을 다시 전역 모델로 전파한다.
            self._broadcast_global()

            mean_client_loss = sum(u["loss"] for u in client_updates) / len(client_updates)
            mean_pseudo = sum(u.get("num_pseudo", 0) for u in client_updates) / len(client_updates)
            self._log(
                "phase1", rnd,
                client_loss=mean_client_loss,
                n_pseudo=mean_pseudo,
                server_loss=server_loss,
                num_selected=len(selected),
                tau_low=tau_low,
                tau_high=tau_high,
            )
            if self.eval_fn and rnd % self.cfg.log_every == 0:
                self.eval_fn(self.global_model, rnd, "phase1")

        self._save_ckpt("phase1")

        # ---------- 2단계: FPT + 직교 보정 ----------
        print("=== Phase 2: Full Parameter Training + Orthogonal ===")
        self.server.reset_optimizer()  # 2단계에서는 새 optimizer로 시작한다.
        for rnd in range(self.cfg.T2):
            progress = (self.cfg.T1 + rnd) / max(total_rounds - 1, 1)
            tau_low, tau_high = self.cfg.get_thresholds(progress)

            # 1단계) 클라이언트가 전체 파라미터 학습 + 가짜 라벨 학습을 수행한다.
            selected = self._sample_clients()
            client_updates = []
            for c in selected:
                upd = c.train_phase2(self.global_model, tau_low=tau_low, tau_high=tau_high)
                client_updates.append(upd)

            # 2단계) 2단계에서는 전체 state_dict를 FedAvg로 합친다.
            agg_full = fedavg(
                [u["state_dict"] for u in client_updates],
                [u["num_samples"] for u in client_updates],
            )
            agg_full = {k: v.to(self.cfg.device) for k, v in agg_full.items()}
            self.global_model.load_state_dict(agg_full)

            # 3단계) 서버가 레이블 지도 보정 + 직교 보정을 수행한다.
            self.server.model.load_state_dict(self.global_model.state_dict())
            server_loss = self.server.update(use_ortho=True)
            self._broadcast_global()

            mean_client_loss = sum(u["loss"] for u in client_updates) / len(client_updates)
            mean_client_ortho = sum(u["ortho"] for u in client_updates) / len(client_updates)
            mean_pseudo = sum(u.get("num_pseudo", 0) for u in client_updates) / len(client_updates)
            self._log(
                "phase2", rnd,
                client_loss=mean_client_loss,
                n_pseudo=mean_pseudo,
                client_ortho=mean_client_ortho,
                server_loss=server_loss,
                tau_low=tau_low,
                tau_high=tau_high,
            )
            if self.eval_fn and rnd % self.cfg.log_every == 0:
                self.eval_fn(self.global_model, rnd, "phase2")

        self._save_ckpt("phase2")
        print("=== FedSTO done ===")
