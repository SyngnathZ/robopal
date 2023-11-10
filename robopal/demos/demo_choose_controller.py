import argparse
import numpy as np
from robopal.robots.diana_med import DianaMed
from robopal.envs import JntCtrlEnv, PosCtrlEnv


parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='JNTIMP', type=str, help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

if args.ctrl == 'JNTIMP':
    env = JntCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        jnt_controller='JNTIMP',
    )
    action = np.array([0.33116, -0.39768533, 0.66947228, 0.33116, -0.39768533, 0.66947228, 0])

elif args.ctrl == 'JNTVEL':
    env = JntCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        jnt_controller='JNTVEL',
    )
    action = np.array([0.01, -0.01, 0.0, 0.0, 0.01, 0.01, 0])

elif args.ctrl == 'OSC':
    env = PosCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        is_pd=False
    )
    action = np.array([0.33116, -0.39768533, 0.66947228])

else:
    env = None
    raise ValueError('Invalid controller type')

env.reset()
for t in range(int(1e6)):
    env.step(action)
    if env.is_render:
        env.render()
