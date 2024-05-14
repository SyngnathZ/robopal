import argparse
import numpy as np
import logging

from robopal.robots.fr5_cobot import DualFR5Cobot
from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='CARTIK', type=str,
                    help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

assert args.ctrl in ['JNTIMP', 'CARTIK'], 'Invalid controller'

if args.ctrl == 'JNTIMP':
    actions = [np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7]),
               np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7])]

elif args.ctrl == 'CARTIK':
    actions = [np.array([0.3, 0.0, 0.5, 0, 1, 0, 0]),
               np.array([0.5, 0.0, 0.5, 0, 1, 0, 0])]
else:
    raise ValueError('Invaild controller.')

env = RobotEnv(
    robot=DualFR5Cobot,
    render_mode='human',
    control_freq=200,
    is_interpolate=False,
    controller=args.ctrl,
)
env.controller.reference = 'world'

actions = {agent: actions[id] for id, agent in enumerate(env.agents)}

env.reset()
for t in range(int(1e4)):
    env.step(actions)
env.close()
