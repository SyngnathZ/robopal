import argparse
import numpy as np
import logging

from robopal.robots.fr5_cobot import FR5Cobot
from robopal.envs import RobotEnv, PosCtrlEnv

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    options = {}

    # Choose controller
    options['ctrl'] = 'CARTIK'

    assert options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    if options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP']:
        env = RobotEnv(
            robot=FR5Cobot(),
            render_mode='human',
            control_freq=200,
            is_interpolate=False,
            controller=options['ctrl'],
        )

        if options['ctrl'] == 'JNTIMP':
            action = np.array([0.2, 0.2, 0, 0, 0, 0])

        elif options['ctrl'] == 'JNTVEL':
            action = np.array([0.01, -0.01, 0.0, 0.0, 0.01, 0.01])

        elif options['ctrl'] == 'CARTIMP':
            action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0.0])

    elif options['ctrl'] == 'CARTIK':
        env = PosCtrlEnv(
            robot=FR5Cobot(),
            render_mode='human',
            control_freq=200,
            is_interpolate=False,
            is_pd=False
        )
        action = np.array([0.33, -0.39, 0.66, 1, 0, 0, 0])
    else:
        raise ValueError('Invalid controller')

    if isinstance(env, RobotEnv):
        env.reset()
        for t in range(int(1e4)):
            env.step(action)
        env.close()
