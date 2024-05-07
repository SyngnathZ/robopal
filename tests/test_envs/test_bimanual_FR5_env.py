import argparse
import numpy as np
import logging

from robopal.robots.fr5_cobot import DualFR5Cobot
from robopal.envs import RobotEnv, PosCtrlEnv

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='JNTIMP', type=str,
                    help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

if args.ctrl not in ['JNTIMP', 'CARTIK']:
    raise ValueError('Invalid controller')

if args.ctrl == 'JNTIMP':
    env = RobotEnv(
        robot=DualFR5Cobot(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=args.ctrl,
    )

    actions = [np.array([0.3, -2.4, -0.7, 0.3, -0.4, 0.7]),
               np.array([0.3, -2.4, -0.7, 0.3, -0.4, 0.7])]

else:  # args.ctrl == 'CARTIK'
    env = PosCtrlEnv(
        robot=DualFR5Cobot(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        is_pd=False
    )
    actions = [np.array([0.3, 0.3, 0.4, 1, 0, 0, 0]),
               np.array([0.4, -0.4, 0.6, 1, 0, 0, 0])]

actions = {agent: actions[id] for id, agent in enumerate(env.agents)}

if isinstance(env, RobotEnv):
    env.reset()
    for t in range(int(1e4)):
        env.step(actions)
    env.close()