# ACT Transformer Reproduction

Handmade PyTorch reproduction of [ACT (Action Chunking Transformer)](https://tonyzhaozh.github.io/aloha/).

## Environment

```bash
conda activate aloha5090
```

Requirements: `torch`, `torchvision`, `numpy`, `h5py`, `dm_control`, `opencv-python`, `matplotlib`.

## Project Structure

```
├── main.py                     # Training + eval + visualization entry point
├── models/
│   ├── transformer.py          # Transformer Encoder/Decoder/Attention/FFN
│   ├── vae.py                  # DETRVAE (CVAE + visual Transformer)
│   ├── vision_encoder.py       # ResNet backbone with FrozenBatchNorm
│   └── position_embedding.py   # 2D sinusoidal position encoding
├── util/
│   ├── config.py               # Task configs, MuJoCo constants, gripper lambdas
│   ├── data_load.py            # HDF5 dataset loader + normalization
│   └── visualize.py            # save_videos (rollout mp4), visualize_joints
├── envs/
│   ├── sim_env.py              # Joint-space MuJoCo simulation environment
│   └── ee_sim_env.py           # End-effector space simulation environment
└── assets/                     # MuJoCo XML models + STL meshes (22 files)
```

---

## 1. Training

Train an ACT policy from demonstration data.

```bash
python3 main.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ./checkpoints/run1 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0
```

### Key training parameters

| Flag | Default | Description |
|---|---|---|
| `--task_name` | *(required)* | One of `sim_transfer_cube_scripted`, `sim_transfer_cube_human`, `sim_insertion_scripted`, `sim_insertion_human` |
| `--ckpt_dir` | *(required)* | Output directory for checkpoints and logs |
| `--num_epochs` | `2000` | Total training epochs |
| `--batch_size` | `8` | Batch size per GPU |
| `--lr` | `1e-5` | Learning rate (AdamW) |
| `--kl_weight` | `10.0` | Weight for KL divergence term in CVAE loss |
| `--chunk_size` | `100` | Number of action queries predicted per chunk |
| `--hidden_dim` | `512` | Transformer hidden dimension |
| `--dim_feedforward` | `3200` | Transformer FFN hidden dimension |
| `--enc_layers` | `4` | Number of encoder layers (posterior + main) |
| `--dec_layers` | `7` | Number of decoder layers |
| `--seed` | `0` | Random seed for reproducibility |

### Advanced options

| Flag | Description |
|---|---|
| `--backbone {resnet18,resnet34,resnet50}` | Vision backbone (default: `resnet18`) |
| `--pre_norm` | Use pre-LayerNorm in encoder (default: post-LN) |
| `--dilation` | Replace stride with dilation in ResNet |
| `--no_augmentation` | Disable ColorJitter during training |
| `--no_pretrained_backbone` | Train backbone from scratch |
| `--resume <path>` | Resume from a checkpoint |
| `--validate_every N` | Validation interval (default: `50`) |
| `--save_every N` | Periodic checkpoint interval (default: `500`) |
| `--weight_decay` | AdamW weight decay (default: `1e-4`) |
| `--grad_clip` | Gradient clipping norm (default: `1.0`) |

### Training outputs

```
checkpoints/run1/
├── policy_best.ckpt        # Best validation loss checkpoint
├── policy_last.ckpt        # Final epoch checkpoint
├── policy_epoch_500.ckpt   # Periodic checkpoints
├── commandline_args.json   # Full CLI arguments for reproducibility
```

### Training pipeline (per batch)

```
Dataloader ([0,1] image) → ColorJitter (p=0.8) → ImageNet Normalize → Model forward
                                                                         ├── Posterior: [CLS, qpos, actions] → (μ, σ) → z
                                                                         ├── Vision: ResNet → 1×1 conv → 2D PE
                                                                         └── Transformer: encoder(z, qpos, pixels) → decoder queries → actions
Loss: L1(actions_pred, actions_gt) + kl_weight × D_KL(N(μ,σ) ‖ N(0,1))
```

---

## 2. Evaluation (Rollout)

Evaluate a trained checkpoint in the MuJoCo simulation environment. Success/failure is determined **automatically** by the physics engine via collision detection — no human annotation needed.

```bash
python3 main.py --eval \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ./checkpoints/run1 \
  --temporal_agg \
  --num_rollouts 50
```

### Evaluation parameters

| Flag | Default | Description |
|---|---|---|
| `--eval` | | Trigger evaluation mode (skips training) |
| `--temporal_agg` | | Enable temporal ensemble for smoother actions |
| `--num_rollouts` | `50` | Number of evaluation episodes |
| `--onscreen_render` | | Show live MuJoCo rendering during rollout |

### How success is determined

The MuJoCo physics engine detects object contacts every timestep. In `TransferCubeTask`:

| Reward | Condition |
|---|---|
| 0 | Nothing happening |
| 1 | Right gripper touches the red cube |
| 2 | Right gripper lifts the cube off the table |
| 3 | Left gripper touches the cube (transfer attempt) |
| 4 | Left gripper lifts the cube (successful transfer) |

**Success** = `highest_reward == 4` for that rollout.

### Evaluation pipeline (per rollout)

```
Reset env (random cube pose)
  ↓
for t = 0..399:
  MuJoCo render → /255 → ImageNet Normalize → Policy(qpos, image) → a_hat
  Temporal ensemble: exp-weight average of overlapping predictions
  Post-process: action × std + mean → env.step(action)
  MuJoCo auto-detects contacts → reward
  ↓
video{id}.mp4 saved
Success rate = N_success / N_rollouts
```

### Temporal ensemble

When `--temporal_agg` is on, the policy is queried **every step** (instead of every `num_queries` steps). Multiple overlapping predictions are combined with exponential weighting — older predictions get higher weight (`exp(-0.01 × i)`). This smooths the action trajectory and significantly improves success rate.

### Evaluation outputs

```
checkpoints/run1/
├── video0.mp4 ... video49.mp4   # Rollout videos (multi-camera concatenated)
├── eval_result.txt               # Success rate breakdown per reward level
```

Example output:
```
Rollout   0  return=  786.0  highest_reward=4  max_reward=4  Success: True
Rollout   1  return=  744.0  highest_reward=4  max_reward=4  Success: True
...

Success rate: 80.00%
Average return: 603.4

Reward >= 0: 5/5 = 100.0%
Reward >= 1: 5/5 = 100.0%
Reward >= 2: 5/5 = 100.0%
Reward >= 3: 4/5 = 80.0%
Reward >= 4: 4/5 = 80.0%
```

---

## 3. Visualization

### 3a. Rollout videos (auto-generated during eval)

Each `--eval` run produces one `.mp4` per rollout. Videos horizontally concatenate all available camera views (`top`, `angle`, `vis`).

The video generation uses `util/visualize.py:save_videos()`:
- Input: list of dicts `[{cam: (H,W,3) uint8}, ...]`
- RGB → BGR conversion (for OpenCV)
- mp4v codec at `1/DT = 50 fps`

### 3b. Joint trajectory plots

`util/visualize.py:visualize_joints()` plots qpos and command trajectories across all 14 joints. Can be called programmatically:

```python
from util.visualize import visualize_joints

visualize_joints(
    qpos_list,        # list of (14,) numpy arrays
    command_list,      # list of (14,) numpy arrays
    plot_path='joints.png',
)
```

### 3c. Standalone dataset visualization

The original ACT also provides `record_sim_episodes.py` for recording and `visualize_episodes.py` for visualizing stored HDF5 episodes. These are in the original `act/` directory and not part of this reproduction repo.

---

## Full Workflow Example

```bash
conda activate aloha5090

# 1. Train the model (2000 epochs)
python3 main.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ./checkpoints/run1 \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0

# 2. Evaluate with temporal ensemble (50 rollouts)
python3 main.py --eval \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ./checkpoints/run1 \
  --temporal_agg --num_rollouts 50

# 3. Check results
cat ./checkpoints/run1/eval_result.txt
ls ./checkpoints/run1/video*.mp4
```

---

## Data Format

Training data is stored as HDF5 files under `datasets/<task_name>/`:

```
datasets/sim_transfer_cube_scripted/
├── episode_0.hdf5
├── episode_1.hdf5
...
└── episode_49.hdf5
```

Each HDF5 file:
```
attrs['sim'] = True
/observations/qpos          [T, 14]  float64
/observations/qvel          [T, 14]  float64
/observations/images/top    [T, 480, 640, 3]  uint8
/action                     [T, 14]  float64
```

## Task Configuration

Available tasks (from `util/config.py`):

| Task | Episodes | Steps | Cameras |
|---|---|---|---|
| `sim_transfer_cube_scripted` | 50 | 400 | `['top']` |
| `sim_transfer_cube_human` | 50 | 400 | `['top']` |
| `sim_insertion_scripted` | 50 | 400 | `['top']` |
| `sim_insertion_human` | 50 | 500 | `['top']` |

---

## Key Design Notes

1. **Image normalization**: Images are always processed as `/255.0 → ImageNet Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`. This is required by the pretrained ResNet backbone and is applied consistently in both training and evaluation.

2. **Position encoding**: DETR-style — position info is added only to Q and K in attention, NOT to V. 2D sine encoding for image features, learned embeddings for query tokens.

3. **CVAE posterior**: During training, the encoder sees `[CLS, qpos, ground-truth actions]` to produce `(μ, σ)`. During inference, `z=0` (prior mean).

4. **Post-LN only**: The decoder uses post-LayerNorm. Pre-LN is available for the encoder but not validated.
