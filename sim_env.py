import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    """
    Environment for simulated robot single-arm manipulation, with joint position control
    Action space:      [arm_qpos (6),                 # absolute joint position
                        gripper_position (1),         # normalized gripper position (0: close, 1: open)
                        body_z (1),                    # body z position
                        body_vel_x (1),                # body velocity x
                        body_vel_y (1),                # body velocity y
                        body_pitch (1)]                # body pitch angle

    Observation space: {"qpos": Concat[ arm_qpos (6),             # absolute joint position
                                        gripper_position (1),     # normalized gripper position (0: close, 1: open)
                                        body_z (1),                # body z position
                                        body_vel_x (1),            # body velocity x
                                        body_vel_y (1),            # body velocity y
                                        body_pitch (1)]            # body pitch angle
                        "qvel": Concat[ arm_qvel (6),             # absolute joint velocity (rad)
                                        gripper_velocity (1),     # normalized gripper velocity (pos: opening, neg: closing)
                                        body_z_vel (1),            # body z velocity
                                        body_vel_x (1),            # body velocity x
                                        body_vel_y (1),            # body velocity y
                                        body_pitch_vel (1)]        # body pitch velocity
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        arm_action = action[:6]
        normalized_gripper_action = action[6]
        body_z = action[7]
        body_vel_x = action[8]
        body_vel_y = action[9]
        body_pitch = action[10]

        gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_gripper_action)
        full_gripper_action = [gripper_action, -gripper_action]

        env_action = np.concatenate([arm_action, full_gripper_action])
        # Note: body_z, body_vel_x, body_vel_y, body_pitch are used for state tracking but don't directly control the arm
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos_raw = qpos_raw[:8]
        arm_qpos = arm_qpos_raw[:6]
        gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(arm_qpos_raw[6])]

        # Get body state (z position, x velocity, y velocity, pitch)
        # Assuming body info is in remaining qpos indices or can be derived from physics
        body_z = qpos_raw[16] if len(qpos_raw) > 16 else 0.0  # body z position
        body_vel_x = 0.0  # default, will be populated from body velocity
        body_vel_y = 0.0  # default, will be populated from body velocity
        body_pitch = 0.0  # default, will be populated from body orientation

        return np.concatenate([arm_qpos, gripper_qpos, [body_z], [body_vel_x], [body_vel_y], [body_pitch]])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel_raw = qvel_raw[:8]
        arm_qvel = arm_qvel_raw[:6]
        gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(arm_qvel_raw[6])]

        # Get body state velocities
        body_z_vel = 0.0  # default, will be populated from body velocity
        body_vel_x = 0.0  # default, will be populated from body velocity
        body_vel_y = 0.0  # default, will be populated from body velocity
        body_pitch_vel = 0.0  # default, will be populated from body angular velocity

        return np.concatenate([arm_qvel, gripper_qvel, [body_z_vel], [body_vel_x], [body_vel_y], [body_pitch_vel]])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['arm_camera'] = physics.render(height=480, width=640, camera_id='arm_camera')
        obs['images']['zed_camera'] = physics.render(height=480, width=640, camera_id='zed_camera')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right=None):
    action = np.zeros(11)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    # gripper action
    gripper_pos = master_bot_left.dxl.joint_states.position[7]
    normalized_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(gripper_pos)
    action[6] = normalized_pos
    # body state action (z position, vel_x, vel_y, pitch)
    action[7] = 0.0  # body_z
    action[8] = 0.0  # body_vel_x
    action[9] = 0.0  # body_vel_y
    action[10] = 0.0  # body_pitch
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

