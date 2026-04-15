from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class OptimConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class FedSTOConfig:
    # 라운드 수 (논문 기준: 총 300 = warmup 50 + T1 100 + T2 150)
    warmup_rounds: int = 50           # 서버 지도 사전학습 라운드 수
    T1: int = 100                     # 1단계(선택적 학습) 라운드 수
    T2: int = 150                     # 2단계(FPT + 직교 보정) 라운드 수
    local_epochs: int = 1             # 라운드당 클라이언트 로컬 에폭 수 (논문 기준 1)
    server_steps: int = 20            # 집계 후 서버 지도학습 스텝 수

    # 연합 샘플링
    num_clients: int = 3              # 논문 기준 3개 (맑음 / 야간 / 악천후)
    client_sample_ratio: float = 1.0  # 1.0이면 매 라운드 모든 클라이언트 참여

    # 로컬 EMA teacher
    ema_decay: float = 0.999
    reset_ema_each_round: bool = True # broadcast된 global로 로컬 EMA를 다시 초기화

    # 가짜 라벨 임계값 (논문 기준: ignore threshold 0.1 ~ 0.6)
    tau_low: float = 0.1              # NMS confidence 임계값 / soft 경계
    tau_high: float = 0.6             # hard pseudo-label 경계

    # Epoch Adaptor: 학습 동안 임계값을 선형으로 조정
    use_epoch_adaptor: bool = True
    tau_low_start: float = 0.1        # 학습 시작 시 tau_low 값 (논문 기준 고정)
    tau_high_start: float = 0.3       # 학습 시작 시 tau_high 값

    # 비지도 손실 가중치
    unsup_loss_weight: float = 4.0

    # 직교 정규화(SRIP)
    ortho_lambda: float = 1e-3
    ortho_power_iters: int = 1        # spectral norm 계산용 power iteration 횟수

    # 옵티마이저
    server_opt: OptimConfig = field(default_factory=OptimConfig)
    client_opt: OptimConfig = field(default_factory=OptimConfig)

    # 실행 환경
    device: str = "cuda"
    seed: int = 42

    # 로깅
    log_every: int = 1
    ckpt_dir: str = "./checkpoints"

    def get_thresholds(self, progress: float) -> Tuple[float, float]:
        """주어진 학습 진행률 [0, 1]에 대한 (tau_low, tau_high)를 반환한다.

        ``use_epoch_adaptor``가 True이면 ``*_start`` 값에서 최종 값까지
        임계값을 선형 보간해 Efficient Teacher의 Epoch Adaptor를 흉내 낸다.
        """
        progress = max(0.0, min(1.0, progress))
        if not self.use_epoch_adaptor:
            return self.tau_low, self.tau_high
        tl = self.tau_low_start + (self.tau_low - self.tau_low_start) * progress
        th = self.tau_high_start + (self.tau_high - self.tau_high_start) * progress
        return tl, th
