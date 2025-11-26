import time
import os
import numpy as np

from bosdyn.client.robot import Robot
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.image import ImageClient

from spot_utils import authenticate, verify_estop
from spot_lease import LeaseClient, LeaseKeepAlive

# ------------------------------------------------------------
# For ACT interface
# ------------------------------------------------------------
class TimeStep:
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward


# ------------------------------------------------------------
# SpotRealEnv
# ------------------------------------------------------------
class SpotRealEnv:
    def __init__(self, hostname, dt=0.05):
        self.dt = dt

        # ------------------------------
        # Connect to Spot
        # ------------------------------
        sdk = create_standard_sdk("spot_act_client")
        self.robot: Robot = sdk.create_robot(hostname)
        authenticate(self.robot)
        verify_estop(self.robot)

        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.lease_client.take()
        self.lease_keepalive = LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        )

        self.robot.time_sync.wait_for_sync()

        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name
        )

        # List of image sources you want (adjust to your Spot config)
        # TODO: change
        self.camera_sources = [
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
        ]

    # ------------------------------------------------------------
    # Reset robot → stow arm + move back to original position
    # ------------------------------------------------------------
    def reset(self):

        # Stow arm
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(stow_cmd)
        time.sleep(1.0)

        # TODO: move spot back to original position in front of drawer

        # Return ACT-style timestep
        return self._make_timestep(0.0)

    # ------------------------------------------------------------
    # Step the real robot
    # action: 11-dim vector (7 arm joints + 4 body params)
    # ------------------------------------------------------------
    def step(self, action):
        assert len(action) == 11, f"Expected 11-dim action, got {len(action)}"

        arm_q = action[:7]
        body_x, body_y, body_z, body_pitch = action[7:]

        # ------------------------------
        # Build arm command
        # ------------------------------
        arm_cmd = RobotCommandBuilder.arm_joint_move_command(
            joint_targets=list(arm_q),
            max_vel=1.0,
            max_acc=1.0
        )

        # ------------------------------
        # Build body pose shift
        # ------------------------------
        body_cmd = RobotCommandBuilder.body_pose_command(
            x=body_x,
            y=body_y,
            z=body_z,
            roll=0.0,
            pitch=body_pitch,
            yaw=0.0,
            frame_name=BODY_FRAME_NAME
        )

        # ------------------------------
        # Combine into one synchronized command
        # ------------------------------
        full_cmd = RobotCommandBuilder.build_synchro_command(arm_cmd, body_cmd)

        # Send command
        self.command_client.robot_command(full_cmd)

        # Sleep for dt to simulate step interval
        # TODO: do we need this?
        time.sleep(self.dt)

        return self._make_timestep(0.0)

    # ------------------------------------------------------------
    # Build ACT timestep: qpos, images
    # ------------------------------------------------------------
    def _make_timestep(self, reward):
        # ------------------------------
        # Arm joint state
        # ------------------------------
        state = self.robot.state_client.get_robot_state()
        arm_state = state.manipulator_state
        mobility_state = state.kinematic_state

        # Spot arm joints
        q_arm = np.array(arm_state.position)  # length 7

        # ------------------------------
        # Body state
        # ------------------------------
        body_frame = mobility_state.transforms_snapshot.child_to_parent_edge_map["body"]
        pos = body_frame.parent_tform_child.position
        rot = body_frame.parent_tform_child.rotation

        # Extract pitch from quaternion
        pitch = rot.pitch

        # Body qpos (4 dims)
        q_body = np.array([pos.x, pos.y, pos.z, pitch])

        # ------------------------------
        # Build ACT observation
        # ------------------------------
        qpos = np.concatenate([q_arm, q_body])

        # Collect all camera images
        images = {src: self.render(src) for src in self.camera_sources}

        obs = dict(
            qpos=qpos,
            images=images,
        )

        return TimeStep(obs, reward)
