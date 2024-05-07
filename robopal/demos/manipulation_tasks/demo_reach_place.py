import numpy as np
from robopal.demos.manipulation_tasks.robot_reach import EndReachEnv
from robopal.robots.diana_med import DianaGrasp
from robopal.robots.fr5_cobot import FR5Grasp


class ReachPlaceEnv(EndReachEnv):

    def __init__(self,
                 robot=FR5Grasp,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
        )
        self.name = 'ReachPlace-v1'

        self.obs_dim = (8,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.pos_max_bound = np.array([0.6, 0.2, 0.37])
        self.pos_min_bound = np.array([0.3, -0.2, 0.12])
        self.grip_max_bound = 0.02
        self.grip_min_bound = -0.01

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        obs[6] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[7] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

        return {
            'observation': obs.copy(),
            'achieved_goal': end_pos.copy(),  # block position
            'desired_goal': self.get_site_pos('goal_site').copy()
        }

    def _get_info(self) -> dict:
        return {
            'is_success': self._is_success(self.get_site_pos('0_grip_site'), self.get_site_pos('goal_site'), th=0.02)}

    def reset_object(self):
        random_x_pos = np.random.uniform(0.35, 0.55)
        random_y_pos = np.random.uniform(-0.15, 0.15)
        # self.set_object_pose('green_block:joint', np.array([random_x_pos, random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        random_goal_x_pos = np.random.uniform(0.35, 0.55)
        random_goal_y_pos = np.random.uniform(-0.15, 0.15)
        random_goal_z_pos = np.random.uniform(0.46, 0.66)

        block_pos = np.array([random_x_pos, random_y_pos, 0.46])
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        while np.linalg.norm(block_pos - goal_pos) <= 0.05:
            random_goal_x_pos = np.random.uniform(0.4, 0.6)
            random_goal_y_pos = np.random.uniform(-0.2, 0.2)
            random_goal_z_pos = np.random.uniform(0.45, 0.66)
            goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        site_id = self.get_site_id('goal_site')
        self.mj_model.site_pos[site_id] = goal_pos


if __name__ == "__main__":
    env = ReachPlaceEnv()
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
