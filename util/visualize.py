"""Visualization utilities for ACT evaluation rollouts.

Adapted from the original ACT ``visualize_episodes.py``.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


def save_videos(video, dt, video_path=None):
    """Save rollout images as an MP4 video.

    Supports two input formats:

    1. ``video`` is a list of dicts (used during eval rollout)::

         [ {cam_name: (H,W,3) uint8 array}, {cam_name: ...}, ... ]

       Cameras are horizontally concatenated.

    2. ``video`` is a dict of arrays (shape ``(T,H,W,3)`` per camera)::

         {cam_name: (T,H,W,3) uint8 array, ...}

       Cameras are horizontally concatenated.
    """
    if video_path is None:
        video_path = os.path.join(os.getcwd(), 'rollout.mp4')

    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for _ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # RGB -> BGR for OpenCV
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')

    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # RGB -> BGR
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None,
                     label_overwrite=None):
    """Plot joint positions and commands over time."""
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)
    command = np.array(command_list)
    h, w = 2, qpos.shape[1]
    fig, axs = plt.subplots(qpos.shape[1], 1, figsize=(w, h * qpos.shape[1]))

    all_names = [f'{name}_left' for name in STATE_NAMES] + \
                [f'{name}_right' for name in STATE_NAMES]

    for dim_idx in range(qpos.shape[1]):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    for dim_idx in range(qpos.shape[1]):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(qpos.shape[1]):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f'Saved qpos plot to: {plot_path}')
    plt.close()
