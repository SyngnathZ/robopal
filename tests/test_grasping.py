import argparse
import numpy as np
import logging

from robopal.robots.ur5e import UR5eGrasp
from robopal.robots.diana_med import DianaGrasp
from robopal.robots.fr5_cobot import FR5Grasp
from robopal.robots.panda import Panda, PandaGrasp
from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    options = {}

    # Choose controller
    options['ctrl'] = 'CARTIK'

    assert options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    env = RobotEnv(
        robot=FR5Grasp,
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=options['ctrl'],
    )
    env.controller.reference = 'world'

    if options['ctrl'] == 'JNTIMP':
        action = np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])

    elif options['ctrl'] == 'JNTVEL':
        action = np.array([0.1, 0.1, 0.0, 0.0, 0., 0., 0])

    elif options['ctrl'] == 'CARTIMP':
        action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

    elif options['ctrl'] == 'CARTIK':
        action = np.array([0.6, 0.2, 0.37, 1, 0, 0, 0])

    def test_JNTIMP_error():
        print(np.linalg.norm(action - env.robot.get_arm_qpos()))
    
    def test_CARTIK_error():
        current_pos, current_quat = env.controller.forward_kinematics(env.robot.get_arm_qpos())
        target_pos, target_quat = env.mj_data.body("green_block").xpos.copy(), env.mj_data.body(
            "green_block").xquat.copy()
        # print("Current Error:" + str(np.sum(np.abs(target_pos - current_pos))))

    def get_desired_pos():
        mocap_pos = env.mj_data.body("green_block").xpos.copy()
        additional_array = np.array([0, 1, 0, 0])
        # print("Current Pos:" + str(mocap_pos))

        return np.concatenate((mocap_pos, additional_array))

    def get_current_qpos():
        pos= env.robot.get_arm_qpos().copy()
        print(pos)

    if isinstance(env, RobotEnv):
        env.reset()
        while True:
            env.step(get_desired_pos())
            get_current_qpos()
            test_CARTIK_error()
        env.close()
