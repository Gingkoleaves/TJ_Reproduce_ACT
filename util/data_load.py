# Load data from hdf5
#
# ACT hdf5 structure:
# attrs['sim']                         bool
# /observations/qpos                  [T, 14]
# /observations/qvel                  [T, 14]
# /observations/images/top            [T, 480, 640, 3]
# /action                             [T, 14]

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Episode_Dataset(Dataset):
    def __init__(self, episode_ids, dataset_path, cameras, norm_stat):
        self.episode_ids = episode_ids
        self.dataset_path = dataset_path
        self.cameras = cameras
        self.norm_stat = norm_stat
        self.sim = None
        self.__getitem__(0)  # initialize self.sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        episode_id = self.episode_ids[idx]
        dataset_file = os.path.join(self.dataset_path, f'episode_{episode_id}.hdf5')

        # The ACT files store observations/action directly at the HDF5 root.
        with h5py.File(dataset_file, 'r') as f:
            original_action_shape = f['/action'].shape
            episode_len = original_action_shape[0]
            start_timestep = np.random.randint(0, episode_len)

            qpos = f['/observations/qpos'][start_timestep]
            images = {}
            for camera in self.cameras:
                images[camera] = f[f'/observations/images/{camera}'][start_timestep]

            actions = f['/action'][start_timestep:]
            actions_len = episode_len - start_timestep
            self.sim = bool(f.attrs['sim'])

        # Pad future actions to [episode_len, action_dim].
        padded_actions = np.zeros(original_action_shape, dtype=np.float32)
        padded_actions[:actions_len] = actions
        is_padded = np.zeros(episode_len, dtype=bool)
        is_padded[actions_len:] = True

        # Keep camera as an explicit dimension: [Ncam, 3, H, W].
        stacked_images = np.stack([images[camera] for camera in self.cameras], axis=0)
        stacked_images = np.moveaxis(stacked_images, -1, 1)

        stacked_images = stacked_images.astype(np.float32) / 255.0
        padded_actions = (padded_actions - self.norm_stat['action_mean']) / self.norm_stat['action_std']
        qpos = (qpos - self.norm_stat['qpos_mean']) / self.norm_stat['qpos_std']

        # Keep the original scaffold return order.
        return (
            torch.tensor(qpos, dtype=torch.float32),
            torch.tensor(stacked_images, dtype=torch.float32),
            torch.tensor(padded_actions, dtype=torch.float32),
            torch.tensor(is_padded, dtype=torch.bool),
        )


def get_norm_stats(dataset_path, num):
    """Compute qpos/action normalization statistics from episode_0 ... episode_(num-1)."""
    qpos_list = []
    action_list = []

    for episode_id in range(num):
        dataset_file = os.path.join(dataset_path, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_file, 'r') as f:
            qpos_list.append(f['/observations/qpos'][:])
            action_list.append(f['/action'][:])

    qpos_array = np.concatenate(qpos_list, axis=0)
    action_array = np.concatenate(action_list, axis=0)
    return {
        'qpos_mean': np.mean(qpos_array, axis=0),
        'qpos_std': np.clip(np.std(qpos_array, axis=0), 1e-2, None),
        'action_mean': np.mean(action_array, axis=0),
        'action_std': np.clip(np.std(action_array, axis=0), 1e-2, None),
    }


def load_dataset(dataset_path, num_episodes, cameras, batch_size, seed=0):
    train_ratio = 0.8
    rng = np.random.RandomState(seed)
    train_indexs = rng.choice(num_episodes, int(num_episodes * train_ratio), replace=False)
    val_indexs = np.setdiff1d(np.arange(num_episodes), train_indexs)

    norm_stats = get_norm_stats(dataset_path, num_episodes)
    train_dataset = Episode_Dataset(train_indexs, dataset_path, cameras, norm_stats)
    val_dataset = Episode_Dataset(val_indexs, dataset_path, cameras, norm_stats)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, norm_stats, train_dataset.sim
