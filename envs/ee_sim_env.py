"""MuJoCo end-effector space simulation environment for ACT evaluation.

Adapted from the original ACT ``ee_sim_env.py``.  Imports constants from
``util.config``.
"""

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from util.config import (DT, PUPPET_GRIPPER_POSITION_CLOSE,
                         PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
                         PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
                         PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN,
                         START_ARM_POSE, XML_DIR)


def _sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def _sample_insertion_pose():
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def make_ee_sim_env(task_name: str):
    """Create an end-effector space simulation environment for *task_name*.

    Action space (16-dim)::

        [left_ee_pose (7), left_gripper (1), right_ee_pose (7), right_gripper (1)]
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20,
                                  control_timestep=DT, n_sub_steps=None,
                                  flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20,
                                  control_timestep=DT, n_sub_steps=None,
                                  flat_observation=False)
    else:
        raise NotImplementedError(f'Unknown task: {task_name}')
    return env


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array(
            [g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        physics.named.data.qpos[:16] = START_ARM_POSE
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE, -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE, -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos,
                               right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel,
                               right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = {}
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['mocap_pose_left'] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Specific tasks
# ---------------------------------------------------------------------------

class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        cube_pose = _sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx:box_start_idx + 7], cube_pose)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            all_contact_pairs.append((name_geom_1, name_geom_2))

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:
            reward = 2
        if touch_left_gripper:
            reward = 3
        if touch_left_gripper and not touch_table:
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        peg_pose, socket_pose = _sample_insertion_pose()

        def id2index(j_id):
            return 16 + (j_id - 16) * 7

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx:peg_start_idx + 7], peg_pose)

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx:socket_start_idx + 7], socket_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            all_contact_pairs.append((name_geom_1, name_geom_2))

        sockets = ["socket-1", "socket-2", "socket-3", "socket-4"]
        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = any(
            (s, "vx300s_left/10_left_gripper_finger") in all_contact_pairs for s in sockets)
        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = any((s, "table") in all_contact_pairs for s in sockets)
        peg_touch_socket = any(("red_peg", s) in all_contact_pairs for s in sockets)
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:
            reward = 1
        if (touch_left_gripper and touch_right_gripper
                and not peg_touch_table and not socket_touch_table):
            reward = 2
        if peg_touch_socket and not peg_touch_table and not socket_touch_table:
            reward = 3
        if pin_touched:
            reward = 4
        return reward
