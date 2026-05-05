"""논문 알고리즘 1 전체를 실제 코드 흐름으로 돌리는 파일
"""
from __future__ import annotations
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
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

    def _blend_into_global(self, agg_sd: dict, use_differential_alpha: bool = False) -> None:
        
        alpha_bb = self.cfg.fedavg_alpha
        alpha_nb = self.cfg.fedavg_alpha_nonbackbone if use_differential_alpha else alpha_bb
        old_sd = self.global_model.state_dict()
        for k in agg_sd:
            is_bb = self.global_model._is_backbone(k)
            alpha = alpha_bb if is_bb else alpha_nb
            agg_sd[k] = (alpha * old_sd[k].float() + (1 - alpha) * agg_sd[k].float()).to(old_sd[k].dtype)
      
        self.global_model.load_state_dict(agg_sd, strict=False)

    def _broadcast_global(self) -> None:
       
        self.global_model.load_state_dict(self.server.model.state_dict())

    def _pull_into_server(self, sd: dict) -> None:
        self.server.model.load_state_dict(sd, strict=False)

    def _global_state_snapshot(self) -> dict:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.global_model.state_dict().items()
        }

    def _run_clients(
        self,
        selected: List[Client],
        phase: int,
        tau_low: float,
        tau_high: float,
    ) -> list[dict]:
        train_name = "train_phase1" if phase == 1 else "train_phase2"
        devices = {str(c.device) for c in selected}

        def train_one(c: Client, source) -> dict:
            device = torch.device(c.device)
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)
            return getattr(c, train_name)(
                source,
                tau_low=tau_low,
                tau_high=tau_high,
            )

        if len(devices) <= 1:
            return [train_one(c, self.global_model) for c in selected]

        global_state = self._global_state_snapshot()
        device_locks = {device: Lock() for device in devices}

        def run_one(c: Client) -> dict:
            # 같은 GPU에 배정된 클라이언트는 동시에 올리지 않아 OOM/경합을 피한다.
            with device_locks[str(c.device)]:
                return train_one(c, global_state)

        with ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = [ex.submit(run_one, c) for c in selected]
            return [f.result() for f in futures]

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

    def load_checkpoint(self, ckpt_path: str) -> None:
        """저장된 체크포인트를 global model과 server에 불러온다."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.global_model.load_state_dict(ckpt["state_dict"])
        self.global_model.to(self.cfg.device)
        self.server.model.load_state_dict(ckpt["state_dict"])
        self.server.model.to(self.cfg.device)
        print(f"  Loaded checkpoint: {ckpt_path}")

    
    def run(self, skip_warmup: bool = False, warmup_ckpt: str | None = None,
            skip_phase1: bool = False, phase1_ckpt: str | None = None) -> None:
        if skip_warmup and warmup_ckpt:
            print("=== Skipping warmup, loading checkpoint ===")
            self.load_checkpoint(warmup_ckpt)
        elif not skip_phase1:
            # ---------- 웜업 ---------- #서버만 레이블 데이터로 학습
            print("=== Warmup (server supervised) ===")
            warm_loss = self.server.warmup()
            self._log("warmup", 0, loss=warm_loss)
            self._broadcast_global()
            self._save_ckpt("warmup")
            if self.eval_fn:
                self.eval_fn(self.global_model, 0, "warmup")

        total_rounds = self.cfg.T1 + self.cfg.T2

        if skip_phase1 and phase1_ckpt:
            print("=== Skipping Phase 1, loading checkpoint ===")
            self.load_checkpoint(phase1_ckpt)
        else:
            # ---------- 1단계: 선택적 학습 ----------
            """
                global_model(1) 1개
                server.model(2) 1개
                client.model 3개
                EMA teacher 3개

                client.model_1 + client.model_2 + client.model_3 => fedavg => global_model => global_model을 server.model(라벨이 있는 모델)에 복사 => server.model이 서버에서 학습

                1. train_phase1()으로 각 클라이언트 백본 학습 후 client_updates에 저장
                2. fedavg()로 client_updates에서 백본 가중치 평균 낸 후 global_model(1) 에 블렌딩해서 업데이트
                3. global_model(1)을 server_gloabal_model(2)로 복사
                4. 서버는 labeled data로 server_global_model(2)로 supervised loss 계산해서 모델 업데이트
                5. 학습된 server_global_model(2)를 global_model(1)로 복사
            """
            print("=== Phase 1: Selective Training ===")
            for rnd in range(self.cfg.T1):

                progress = rnd / max(total_rounds - 1, 1)
                tau_low, tau_high = self.cfg.get_thresholds(progress)

                selected = self._sample_clients() #이번 라운드 참여 클라이언트 목록
                client_updates = self._run_clients(selected, phase=1, tau_low=tau_low, tau_high=tau_high)

                # 클라이언트 샘플링 결과를 평균내어 백본을 집계한다.
                agg_backbone = fedavg( 
                    [u["backbone_state_dict"] for u in client_updates], #각 클라이언트가 보낸 백본 가중치 3개를 꺼냄
                    [u["num_samples"] for u in client_updates], #각 클라이언트 데이터 개수를 가중치로 사용
                    exclude_bn=self.cfg.fedavg_exclude_bn, 
                )
                # 평균낸 백본을 블렌딩해 글로벌 모델에 반영한다.
                agg_backbone = {k: v.to(self.cfg.device) for k, v in agg_backbone.items()}
                self._blend_into_global(agg_backbone)

                # 서버가 레이블 데이터로 한 번 더 보정
                self.server.model.load_state_dict(self.global_model.state_dict())  #self.global_model : 클라이언트 업데이트(fedavg)가 반영된 현재 글로벌 모델 , self.server.model : 서버가 레이블 데이터로 지도학습 할 때 쓰는 모델 #global_model을 서버의 model에 복사사
                server_loss = self.server.update(use_ortho=False) #서버가 레이블 데이터로 지도학습 (내가 설정한 서버 보정 step수만큼)
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

        # ---------- 2단계: 전체 파라미터 학습 + 직교 강화 ----------
        print("=== Phase 2: Full Parameter Training + Orthogonal ===")
        self.server.reset_optimizer()  # Phase 2용으로 optimizer를 새로 시작
        for rnd in range(self.cfg.T2):
            progress = (self.cfg.T1 + rnd) / max(total_rounds - 1, 1)
            tau_low, tau_high = self.cfg.get_thresholds(progress)

            selected = self._sample_clients()
            client_updates = self._run_clients(selected, phase=2, tau_low=tau_low, tau_high=tau_high)

            # 전체 파라미터를 집계
            agg_full = fedavg(
                [u["state_dict"] for u in client_updates],
                [u["num_samples"] for u in client_updates],
                exclude_bn=self.cfg.fedavg_exclude_bn,
            )
            # alpha blending으로 글로벌 모델에 반영
            agg_full = {k: v.to(self.cfg.device) for k, v in agg_full.items()}
            self._blend_into_global(agg_full, use_differential_alpha=True)

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
