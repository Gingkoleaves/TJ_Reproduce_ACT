"""MuJoCo simulation environment for ACT evaluation (joint-space control).

Adapted from the original ACT ``sim_env.py``.  Imports constants from
``util.config`` instead of the standalone ``constants`` module.
"""

import collections
import os

import matplotlib.pyplot as plt
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from util.config import (DT, MASTER_GRIPPER_POSITION_NORMALIZE_FN,
                         PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
                         PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
                         PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN,
                         START_ARM_POSE, XML_DIR)

BOX_POSE = [None]  # mutated from outside before env.reset()


def make_sim_env(task_name: str):
    """Create a joint-space simulation environment for *task_name*.

    Action space (14-dim)::

        [left_arm_qpos (6), left_gripper (1), right_arm_qpos (6), right_gripper (1)]

    Observation space::

        {"qpos": 14, "qvel": 14, "images": {"top": (480,640,3), …}}
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20,
                                  control_timestep=DT, n_sub_steps=None,
                                  flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20,
                                  control_timestep=DT, n_sub_steps=None,
                                  flat_observation=False)
    else:
        raise NotImplementedError(f'Unknown task: {task_name}')
    return env


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([
            left_arm_action, full_left_gripper_action,
            right_arm_action, full_right_gripper_action,
        ])
        super().before_step(env_action, physics)

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
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Specific tasks
# ---------------------------------------------------------------------------

class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
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


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2:] = BOX_POSE[0]
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

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        sockets = ["socket-1", "socket-2", "socket-3", "socket-4"]
        touch_left_gripper = any(
            (s, "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            for s in sockets
        )
        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = any((s, "table") in all_contact_pairs for s in sockets)
        peg_touch_socket = any(
            ("red_peg", s) in all_contact_pairs for s in sockets
        )
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
