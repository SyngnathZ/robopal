from robopal.commons.gym_wrapper import GoalEnvWrapper
from robopal.demos.manipulation_tasks.demo_reach_place import ReachPlaceEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
from sb3_contrib import TQC
import os
import argparse

# Create directories to hold models and logs
model_dir = "models/ReachPlaceSingle-Sparse-v3.1"
log_dir = "logs/ReachPlaceSingle-Sparse-v3.1"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    match sb3_algo:
        case 'SAC':
            model = SAC('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'PPO':
            model = PPO('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TQC':
            model = TQC('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")


def test(env, sb3_algo, path_to_model):
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case 'TQC':
            model = TQC.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    env = ReachPlaceEnv()
    env = GoalEnvWrapper(env)
    env.reset()

    if args.train:
        gymenv = env
        train(gymenv, args.sb3_algo)

    if (args.test):
        if os.path.isfile(args.test):
            gymenv = env
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
