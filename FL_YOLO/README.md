# FedSTO

Detector-agnostic implementation of **FedSTO** (Kim et al., NeurIPS 2023):
Semi-Supervised Federated Object Detection with Selective Training and
Orthogonal Enhancement.

## Install

```bash
pip install torch yolov5
```

## Run

**Framework sanity check (DummyDetector, synthetic data, ~10 sec):**

```bash
python -m FL_YOLO.main
```

**Real YOLOv5 end-to-end (synthetic YOLO-format data, ~1 min on CPU):**

```bash
python -m FL_YOLO.run_SSLAD2D
```

### Run BDD100K with 2 GPUs

프로젝트 루트에서 실행합니다.

```bash
cd /home/pyh/바탕화면/FL_GIT
```

먼저 PyTorch가 GPU 2개를 보는지 확인합니다.

```bash
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

BDD100K 전체 학습은 다음처럼 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m FL_YOLO.run_bdd100k
```

실행 초반에 아래처럼 출력되면 클라이언트가 GPU 2개에 자동 배정된 것입니다.

```text
device = cuda
client devices = ['cuda:0', 'cuda:1']
```

처음에는 짧은 스모크 테스트로 2 GPU 동작만 확인하는 것을 권장합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m FL_YOLO.run_bdd100k \
  --warmup-rounds 1 \
  --t1 1 \
  --t2 1 \
  --max-per-client 100 \
  --max-server-images 100 \
  --batch-size 8 \
  --num-workers 2
```

다른 터미널에서 GPU 사용률을 확인합니다.

```bash
watch -n 0.5 nvidia-smi
```

참고:

- `CUDA_VISIBLE_DEVICES=0,1`은 실제 GPU 0번과 1번만 이 프로세스에 보이게 합니다.
- `CUDA_VISIBLE_DEVICES=2,3`처럼 실행하면 코드 안에서는 여전히 `cuda:0`, `cuda:1`로 보이지만, 실제로는 물리 GPU 2번과 3번을 씁니다.
- 클라이언트는 server/global 모델이 있는 `cuda:0` 부담을 줄이기 위해 `cuda:1`, `cuda:0`, `cuda:1` 순서로 라운드로빈 배정됩니다.
- 같은 GPU에 배정된 클라이언트는 동시에 올리지 않고 순서대로 실행해서 OOM 위험을 줄입니다.
- OOM이 나면 `--batch-size`를 먼저 낮추고, CPU 병목이 있으면 `--num-workers`를 낮춰 봅니다.

SSLAD-2D도 같은 방식으로 실행할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m FL_YOLO.run_SSLAD2D
```

Expected output:

```
=== Warmup (server supervised) ===
[warmup][r000] loss=4.5030
=== Phase 1: Selective Training ===
[phase1][r000] client_loss=8.81 n_pseudo=548.0 server_loss=3.43 num_selected=2
[phase1][r001] client_loss=8.20 n_pseudo=629.0 server_loss=2.67 num_selected=2
[phase1][r002] client_loss=5.65 n_pseudo=363.0 server_loss=2.50 num_selected=2
=== Phase 2: Full Parameter Training + Orthogonal ===
[phase2][r000] client_loss=4.28 n_pseudo=288.0 client_ortho=47.29 server_loss=2.51
[phase2][r001] client_loss=5.14 n_pseudo=379.5 client_ortho=47.38 server_loss=2.84
[phase2][r002] client_loss=9.81 n_pseudo=601.0 client_ortho=47.17 server_loss=2.34
=== FedSTO done ===
```

`n_pseudo > 0` confirms the local EMA teacher → NMS → τ_high pseudo-label
pipeline is actually producing targets and flowing gradients through the
student's `ComputeLoss`.

## Package layout

```
FL_YOLO/
├── config.py              # FedSTOConfig: all hyperparameters in one place
├── detector.py            # BaseDetector (ABC) + DummyDetector
├── ema.py                 # LocalEMA (teacher pseudo labeler)
├── orthogonal.py          # SRIP: σ(WWᵀ−I) + σ(WᵀW−I) via power iteration
├── aggregator.py          # FedAvg (full + backbone-only)
├── client.py              # Client.train_phase1 / train_phase2
├── server.py              # Server.warmup / update(use_ortho=...)
├── orchestrator.py        # FedSTO.run() — Algorithm 1 main loop
├── main.py                # framework sanity check (DummyDetector)
├── run_SSLAD2D.py        # SSLAD-2D YOLOv5 entry point
├── yolov5_detector.py     # YOLOv5Detector adapter (ultralytics yolov5)
├── yolo_dataset_SSLAD2D.py # SSLAD-2D dataset + synthetic helpers
├── run_bdd100k.py        # BDD100K YOLOv5 entry point
├── yolo_dataset_bdd100k.py    # BDD100K dataset loaders
```

## Algorithm 1 mapping

| Paper step | Code location |
|---|---|
| Warmup: server supervised on labeled data | `Server.warmup()` |
| **Phase 1 (Selective Training)** | `orchestrator.run()` inner loop over `T1` |
| &nbsp;&nbsp; Client backbone update, neck/head frozen | `Client.train_phase1()` + `BaseDetector.freeze_non_backbone()` |
| &nbsp;&nbsp; Local EMA pseudo labeler, re-init from global | `LocalEMA.reset_from()` |
| &nbsp;&nbsp; Backbone-only aggregation | `fedavg(backbone_state_dicts, ...)` |
| &nbsp;&nbsp; Server fine-tune on labeled data | `Server.update(use_ortho=False)` |
| **Phase 2 (FPT + Orthogonal Enhancement)** | inner loop over `T2` |
| &nbsp;&nbsp; All parameters trained | `Client.train_phase2()` |
| &nbsp;&nbsp; Orthogonal reg on non-backbone | `srip_penalty(model.non_backbone_weight_matrices())` |
| &nbsp;&nbsp; Full state dict aggregation | `fedavg(full_state_dicts, ...)` |
| &nbsp;&nbsp; Server fine-tune with ortho | `Server.update(use_ortho=True)` |

## Swapping in real data (BDD100K / Cityscapes / SODA10M)

Use the dataset module that matches your data (`yolo_dataset_SSLAD2D.py` for SSLAD-2D, `yolo_dataset_bdd100k.py` for BDD100K) or add a new dataset-specific module. Build a `Dataset` that yields:

```python
# labeled sample (server)
{"images": Tensor(3, H, W), "targets": Tensor(k, 5)}  # [cls, cx, cy, w, h]

# unlabeled sample (client)
{"images": Tensor(3, H, W)}
```

and keep using `labeled_yolo_collate` / `unlabeled_yolo_collate`. For BDD100K
weather splits, filter by the `attributes.weather` field in the JSON
annotations and hand each subset to a separate client.

## Swapping in a different detector

Subclass `BaseDetector` and implement:

1. `supervised_loss(images, targets) -> dict[str, Tensor]`
2. `unsupervised_loss(images, teacher) -> dict[str, Tensor]`
3. Override `_is_backbone(name)` / `_is_non_backbone(name)` to match your
   parameter naming scheme (see `YOLOv5Detector` for an example handling flat
   layer indices like `yolo.model.0.conv.weight`).
4. Yield Conv/Linear weights from `non_backbone_weight_matrices()` for the
   SRIP regularizer.

Everything else (FL loop, selective training, aggregation, EMA teacher,
ortho reg) is detector-agnostic.

## Key hyperparameters (`config.py`)

| Knob | Meaning | Default |
|---|---|---|
| `warmup_steps` | Server supervised pretraining steps | 200 |
| `T1`, `T2` | Phase 1 / Phase 2 federated rounds | 50 / 50 |
| `local_steps` | Client gradient steps per round | 20 |
| `server_steps` | Server supervised steps after each aggregation | 20 |
| `ortho_lambda` | SRIP weight (λ in the paper) | 1e-4 |
| `ema_decay` | Local EMA α | 0.999 |
| `client_sample_ratio` | Fraction of clients sampled per round | 1.0 |
| `reset_ema_each_round` | Re-init local EMA from global at round start | True |

## Notes for real runs

1. **Pretrained weights**: COCO-pretrained weights are now loaded
   automatically by default (`pretrained=True` in `YOLOv5Detector`).
   Pass `--no-pretrained` to `run_SSLAD2D.py` to disable.
   Default thresholds (`tau_high=0.5`, `tau_low=0.05`) are tuned for
   pretrained initialization per the paper.
2. **BatchNorm in aggregation**: `fedavg` averages float buffers (BN
   running stats) and copies non-float buffers (`num_batches_tracked`) from
   the first client. For FedBN-style behavior, filter `bn` keys from the
   aggregated state dict.
3. **`ortho_lambda` scaling**: SRIP grows with layer count. For YOLOv5n
   (~10 non-backbone Conv layers) 1e-5 is a safer starting point than the
   1e-4 default. Retune for your backbone size.
4. **`tau_high`**: paper uses high/low thresholds of the Semi-Efficient
   Teacher. Start with `tau_high=0.5`, `tau_low=0.05` for pretrained weights;
   lower `tau_high` to ~0.1 only during cold starts from random init.
