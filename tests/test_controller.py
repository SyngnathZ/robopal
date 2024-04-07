import argparse
import numpy as np
import logging

from robopal.robots.diana_med import DianaMed
from robopal.envs import RobotEnv, PosCtrlEnv

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='CARTIMP', type=str,
                    help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

assert args.ctrl in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK']

if args.ctrl in ['JNTIMP', 'JNTVEL', 'CARTIMP']:
    env = RobotEnv(
        robot=DianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=args.ctrl,
    )

    if args.ctrl == 'JNTIMP':
        action = np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])

    elif args.ctrl == 'JNTVEL':
        action = np.array([0., -0.00, 0.0, 0.0, 0.0, 0.0, 0])

    elif args.ctrl == 'CARTIMP':
        action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

elif args.ctrl == 'CARTIK':
    env = PosCtrlEnv(
        robot=DianaMed(),
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
