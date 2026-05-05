from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class OptimConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class FedSTOConfig:
   
    warmup_rounds: int = 30          
    T1: int = 70                     
    T2: int = 120                     
    local_epochs: int = 1      
    #server_epoch이 true면 무조건 에폭으로 돌고 false일 때 server_steps 수만큼 step으로 돈다. 논문에서는 서버도 매 라운드마다 1 epoch씩 돌았다고 명시되어 있음.       
    server_epoch: bool = False
    server_steps: int = 20            

    # 연합 샘플링 설정
    num_clients: int = 3              # 맑음 / 야간 / 악천후
    client_sample_ratio: float = 1.0  # 1.0이면 매 라운드 모든 클라이언트 참여

    # 로컬 EMA 교사 모델 설정
    ema_decay: float = 0.999
    reset_ema_each_round: bool = True # 논문에서는 매 라운드마다 EMA 초기화

    # 의사 라벨 임계값 설정
    tau_low: float = 0.1              # NMS confidence 임계값 / soft 경계
    tau_high: float = 0.6             # hard pseudo-label 경계

    # 에폭 어댑터: 학습 동안 임계값을 선형적으로 조정
    use_epoch_adaptor: bool = True
    tau_low_start: float = 0.1        # 학습 시작 시 tau_low
    tau_high_start: float = 0.2       # 초기 pseudo collapse를 막기 위해 더 완화된 값으로 시작

    # 비지도 손실 가중치
    unsup_loss_weight: float = 4.0

    # 1단계 변형 실험 설정
    phase1_hard_only: bool = False
    phase1_soft_weight: float = 1.0
    phase1_client_lr_scale: float = 1.0
    phase1_max_batches_per_epoch: int | None = None

    # 연합 평균 설정
    fedavg_alpha: float = 0.7         # backbone: global = alpha * old + (1-alpha) * FedAvg
    fedavg_alpha_nonbackbone: float = 0.95  # neck/head는 Phase 2에서 더 보수적으로 반영
    fedavg_exclude_bn: bool = True    # FedAvg에서 BN running stats 제외

    # 직교 정규화 설정
    ortho_lambda: float = 1e-4
    ortho_power_iters: int = 1        # spectral norm 근사용 power iteration 횟수

    # 옵티마이저 설정
    server_opt: OptimConfig = field(default_factory=OptimConfig)
    client_opt: OptimConfig = field(default_factory=OptimConfig)

    # 실행 환경
    device: str = "cuda"
    seed: int = 42

    # 로그 설정
    log_every: int = 1
    ckpt_dir: str = "./checkpoints"

    def get_thresholds(self, progress: float) -> Tuple[float, float]:
        
        progress = max(0.0, min(1.0, progress))
        if not self.use_epoch_adaptor:
            return self.tau_low, self.tau_high
        tl = self.tau_low_start + (self.tau_low - self.tau_low_start) * progress
        th = self.tau_high_start + (self.tau_high - self.tau_high_start) * progress
        return tl, th
