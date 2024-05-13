import mujoco
import numpy as np
import logging

from robopal.demos.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class EndReachEnv(ManipulateEnv):
    """
    This environment is using for reaching a randomly appeared target position with the end effector.
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=None,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 is_interpolate=False,
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
        )

    def compute_rewards(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = None, **kwargs):
        """ Compute the reward for the task."""
        d = goal_distance(achieved_goal, desired_goal)

        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        if kwargs:
            return -(d >= kwargs['th']).astype(np.float64)
        return -(d >= 0.05).astype(np.float64)

        # """ Dense Reward: the returned reward is the negative Euclidean distance between the block and the goal.
        # """
        # return -d
