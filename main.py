"""ACT (Action Chunking Transformer) training and evaluation script.

Training:
    python3 main.py --task_name sim_transfer_cube_scripted \\
        --ckpt_dir <dir> --policy_class ACT --kl_weight 10 --chunk_size 100 \\
        --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \\
        --num_epochs 2000 --lr 1e-5 --seed 0

Evaluation:
    python3 main.py --eval --task_name sim_transfer_cube_scripted \\
        --ckpt_dir <dir> --temporal_agg
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from envs import BOX_POSE, make_sim_env, sample_box_pose, sample_insertion_pose
from models.vae import build_vae
from util.config import DT, SIM_TASK_CONFIGS
from util.data_load import load_dataset
from util.visualize import save_videos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_augmentation() -> transforms.Compose:
    """ColorJitter matching original ACT: applied with p=0.8 per sample."""
    return transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ], p=0.8),
    ])


# ImageNet normalization -- required by the pretrained ResNet backbone.
# Applied AFTER ColorJitter (which expects [0,1]) and BEFORE the model.
IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ACT Training')

    # ---- Task & paths ---------------------------------------------------
    parser.add_argument('--task_name', type=str, required=True,
                        choices=list(SIM_TASK_CONFIGS.keys()),
                        help='Simulation task name')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--policy_class', type=str, default='ACT',
                        choices=['ACT'])

    # ---- Model architecture ----------------------------------------------
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Number of action queries (num_queries)')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pre_norm', action='store_true',
                        help='Use pre-LayerNorm instead of post-LayerNorm')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--dilation', action='store_true',
                        help='Use dilation in ResNet backbone')
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='Do not load pretrained backbone weights')

    # ---- Training --------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # ---- Augmentation ----------------------------------------------------
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable ColorJitter during training')

    # ---- Checkpointing ---------------------------------------------------
    parser.add_argument('--validate_every', type=int, default=50,
                        help='Run validation every N epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save periodic checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # ---- Evaluation -------------------------------------------------------
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation (rollout) instead of training')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Use temporal ensemble during evaluation')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='Show live environment rendering during rollout')
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Number of evaluation rollouts')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
    num_queries: int,
    augmentation: transforms.Compose | None,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_losses, l1_losses, kl_losses = [], [], []

    for batch in loader:
        qpos, images, actions, is_pad = [b.to(device) for b in batch]

        # ColorJitter on [0,1], then ImageNet normalize (required by ResNet backbone)
        if augmentation is not None:
            B, Ncam, C, H, W = images.shape
            images = images.view(B * Ncam, C, H, W)
            images = augmentation(images)
            images = images.view(B, Ncam, C, H, W)
        images = IMAGENET_NORM(images)

        optimizer.zero_grad()

        actions_pred, _is_pad_pred, (mu, logvar) = model(
            qpos, images, None, actions, is_pad,
        )

        # L1 action reconstruction loss, masked by real (unpadded) steps
        actions_gt = actions[:, :num_queries]
        is_pad_gt = is_pad[:, :num_queries].bool()
        l1_all = F.l1_loss(actions_pred, actions_gt, reduction='none')
        valid_mask = (~is_pad_gt).float().unsqueeze(-1)  # (B, num_queries, 1)
        l1 = (l1_all * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # KL divergence: D_KL(N(mu,sigma) || N(0,1))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        loss = l1 + kl_weight * kl
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_losses.append(loss.item())
        l1_losses.append(l1.item())
        kl_losses.append(kl.item())

    return {
        'total': float(np.mean(total_losses)),
        'l1': float(np.mean(l1_losses)),
        'kl': float(np.mean(kl_losses)),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    kl_weight: float,
    num_queries: int,
) -> dict[str, float]:
    model.eval()
    total_losses, l1_losses, kl_losses = [], [], []

    for batch in loader:
        qpos, images, actions, is_pad = [b.to(device) for b in batch]
        images = IMAGENET_NORM(images)

        actions_pred, _is_pad_pred, (mu, logvar) = model(
            qpos, images, None, actions, is_pad,
        )

        actions_gt = actions[:, :num_queries]
        is_pad_gt = is_pad[:, :num_queries].bool()
        l1_all = F.l1_loss(actions_pred, actions_gt, reduction='none')
        valid_mask = (~is_pad_gt).float().unsqueeze(-1)
        l1 = (l1_all * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = l1 + kl_weight * kl

        total_losses.append(loss.item())
        l1_losses.append(l1.item())
        kl_losses.append(kl.item())

    return {
        'total': float(np.mean(total_losses)),
        'l1': float(np.mean(l1_losses)),
        'kl': float(np.mean(kl_losses)),
    }


# ---------------------------------------------------------------------------
# Evaluation (rollout in MuJoCo simulation)
# ---------------------------------------------------------------------------

def _get_image(ts, camera_names: list[str], device: torch.device) -> torch.Tensor:
    """Extract and stack images from an environment timestep.

    Returns (1, Ncam, 3, H, W) float tensor in [0, 1].
    """
    images = []
    for cam_name in camera_names:
        img = ts.observation['images'][cam_name]        # (H, W, 3) uint8
        img = np.moveaxis(img, -1, 0)                    # (3, H, W)
        images.append(img)
    stacked = np.stack(images, axis=0)                   # (Ncam, 3, H, W)
    return torch.from_numpy(stacked).float().div(255.0).unsqueeze(0).to(device)


def eval_bc(args: argparse.Namespace, ckpt_path: str) -> tuple[float, float]:
    """Run *num_rollouts* evaluation episodes and return (success_rate, avg_return)."""
    set_seed(1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']
    max_timesteps = task_config['episode_len']
    num_queries = args.num_queries

    # ---- Build & load model ----------------------------------------------
    model = build_vae(args).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')

    # ---- Load normalization stats ----------------------------------------
    norm_stats = checkpoint['norm_stats']
    pre_process = lambda q: (q - norm_stats['qpos_mean']) / norm_stats['qpos_std']
    post_process = lambda a: a * norm_stats['action_std'] + norm_stats['action_mean']

    # ---- Create environment ----------------------------------------------
    env = make_sim_env(args.task_name)
    env_max_reward = env.task.max_reward

    query_frequency = 1 if args.temporal_agg else num_queries

    episode_returns: list[float] = []
    highest_rewards: list[int] = []

    for rollout_id in range(args.num_rollouts):
        # Randomise object pose
        if 'sim_transfer_cube' in args.task_name:
            BOX_POSE[0] = sample_box_pose()
        elif 'sim_insertion' in args.task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()

        if args.onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(
                height=480, width=640, camera_id='angle'))
            plt.ion()

        # Temporal ensemble buffers
        if args.temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, args.state_dim],
                device=device,
            )

        image_list: list[dict] = []
        rewards: list[float] = []

        with torch.inference_mode():
            for t in range(max_timesteps):
                if args.onscreen_render:
                    img = env._physics.render(height=480, width=640, camera_id='angle')
                    plt_img.set_data(img)
                    plt.pause(DT)

                # Record observations
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                curr_image = _get_image(ts, camera_names, device)

                # Query policy
                if t % query_frequency == 0:
                    all_actions, _, _ = model(  # (1, K, 14)
                        qpos, IMAGENET_NORM(curr_image), None)

                if args.temporal_agg:
                    all_time_actions[t, t:t + num_queries] = all_actions[0]
                    actions_for_curr_step = all_time_actions[:, t]
                    populated = torch.all(actions_for_curr_step != 0, dim=1)
                    actions_for_curr_step = actions_for_curr_step[populated]
                    k = 0.01
                    weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    weights = weights / weights.sum()
                    weights = torch.from_numpy(weights).to(device).unsqueeze(1)
                    raw_action = (actions_for_curr_step * weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                # Post-process and step
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                ts = env.step(action)
                rewards.append(ts.reward)

        if args.onscreen_render:
            plt.close()

        episode_return = np.sum([r for r in rewards if r is not None])
        episode_highest = int(np.max(rewards))
        episode_returns.append(episode_return)
        highest_rewards.append(episode_highest)

        success = episode_highest == env_max_reward
        print(f'Rollout {rollout_id:3d}  '
              f'return={episode_return:6.1f}  '
              f'highest_reward={episode_highest}  '
              f'max_reward={env_max_reward}  '
              f'Success: {success}')

        # Save rollout video
        video_path = os.path.join(args.ckpt_dir, f'video{rollout_id}.mp4')
        save_videos(image_list, DT, video_path=video_path)

    # ---- Summary ---------------------------------------------------------
    success_rate = float(np.mean(np.array(highest_rewards) == env_max_reward))
    avg_return = float(np.mean(episode_returns))

    summary = f'\nSuccess rate: {success_rate:.2%}\nAverage return: {avg_return:.1f}\n\n'
    for r in range(env_max_reward + 1):
        count = int((np.array(highest_rewards) >= r).sum())
        summary += f'Reward >= {r}: {count}/{args.num_rollouts} = {count / args.num_rollouts * 100:.1f}%\n'

    print(summary)

    # Save results to file
    result_path = os.path.join(args.ckpt_dir, 'eval_result.txt')
    with open(result_path, 'w') as f:
        f.write(summary)
        f.write(f'\nepisode_returns: {episode_returns}\n')
        f.write(f'highest_rewards: {highest_rewards}\n')

    return success_rate, avg_return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    set_seed(args.seed)

    # ---- Resolve task config ---------------------------------------------
    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']
    num_episodes = task_config['num_episodes']

    # Inject derived attributes into args -- build_vae reads from the namespace.
    args.camera_names = camera_names
    args.num_queries = args.chunk_size
    args.state_dim = 14
    args.feature_size = (15, 20)
    args.pretrained_backbone = not args.no_pretrained_backbone

    # ---- Device & paths --------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = Path(args.ckpt_dir)

    # ---- Eval mode -------------------------------------------------------
    if args.eval:
        ckpt_path = os.path.join(args.ckpt_dir, 'policy_best.ckpt')
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(args.ckpt_dir, 'policy_last.ckpt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f'No checkpoint found in {args.ckpt_dir}. '
                f'Expected policy_best.ckpt or policy_last.ckpt')
        eval_bc(args, ckpt_path)
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save full CLI invocation for reproducibility
    with open(ckpt_dir / 'commandline_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ---- Data ------------------------------------------------------------
    dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets', args.task_name)
    train_loader, val_loader, norm_stats, is_sim = load_dataset(
        dataset_path=dataset_dir,
        num_episodes=num_episodes,
        cameras=camera_names,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f'Dataset: {args.task_name}  sim={is_sim}')
    print(f'  Train episodes: {len(train_loader.dataset)}  '
          f'Val episodes: {len(val_loader.dataset)}')

    # ---- Model -----------------------------------------------------------
    model = build_vae(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}  Device: {device}')
    print(f'  hidden_dim={args.hidden_dim}  dim_feedforward={args.dim_feedforward}  '
          f'enc_layers={args.enc_layers}  dec_layers={args.dec_layers}  '
          f'nheads={args.nheads}  num_queries={args.num_queries}')

    # ---- Optimizer & scheduler -------------------------------------------
    backbone_param_ids = {id(p) for n, p in model.named_parameters() if 'backbone' in n}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': other_params},
        {'params': backbone_params, 'lr': args.lr * 0.1},
    ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01,
    )

    # ---- Augmentation ----------------------------------------------------
    augmentation = None if args.no_augmentation else get_augmentation()

    # ---- Resume ----------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}  best_val_loss={best_val_loss:.6f}')

    # ---- Training loop ---------------------------------------------------
    print(f'\nTraining for {args.num_epochs} epochs '
          f'(validate every {args.validate_every}, save every {args.save_every})')
    print(f'kl_weight={args.kl_weight}  lr={args.lr}  batch_size={args.batch_size}')
    print(f'Augmentation: {augmentation is not None}')
    print('-' * 72)

    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.kl_weight, args.num_queries, augmentation, args.grad_clip,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch:4d}/{args.num_epochs}  '
              f'Train | L1: {train_metrics["l1"]:.6f}  '
              f'KL: {train_metrics["kl"]:.6f}  '
              f'Total: {train_metrics["total"]:.6f}  '
              f'LR: {current_lr:.2e}')

        if (epoch + 1) % args.validate_every == 0:
            val_metrics = validate(
                model, val_loader, device,
                args.kl_weight, args.num_queries,
            )
            print(f'Epoch {epoch:4d}/{args.num_epochs}  '
                  f' Val  | L1: {val_metrics["l1"]:.6f}  '
                  f'KL: {val_metrics["kl"]:.6f}  '
                  f'Total: {val_metrics["total"]:.6f}')

            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'norm_stats': norm_stats,
                    'args': vars(args),
                }, ckpt_dir / 'policy_best.ckpt')
                print(f'  -> Best checkpoint saved (val_total={best_val_loss:.6f})')

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'norm_stats': norm_stats,
                'args': vars(args),
            }, ckpt_dir / f'policy_epoch_{epoch+1}.ckpt')

    # ---- Final checkpoint ------------------------------------------------
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'norm_stats': norm_stats,
        'args': vars(args),
    }, ckpt_dir / 'policy_last.ckpt')

    print('-' * 72)
    print(f'Training complete.  Best validation loss: {best_val_loss:.6f}')
    print(f'Checkpoints saved to: {ckpt_dir.resolve()}')


if __name__ == '__main__':
    main()
