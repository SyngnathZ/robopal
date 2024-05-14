import numpy as np
from robopal.demos.manipulation_tasks.dualRobot_reach import DualEndReachEnv
from robopal.robots.fr5_cobot import DualFR5Cobot


class DualReachPlaceEnv(DualEndReachEnv):

    def __init__(self,
                 robot=DualFR5Cobot,
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
        self.name = 'DualFR5ReachPlace-v1'

        self.obs_dim = (16,)
        self.goal_dim = (6,)
        self.action_dim = (8,)

        self.max_action = 1.0
        self.min_action = -1.0

        # Adjust 1_grip_site to green color for visualization
        site_id_grip_1 = self.get_site_id('1_grip_site')
        self.mj_model.site_rgba[site_id_grip_1] = [0, 1, 0, 1]

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)
        desired_goal = {agent: 0 for agent in self.robot.agents}
        """ Batch formatting for multi-agent environments."""
        end_pos = desired_goal.copy()
        end_vel = desired_goal.copy()

        end_pos['arm0'] = self.get_site_pos('0_grip_site')
        end_pos['arm1'] = self.get_site_pos('1_grip_site')

        end_vel['arm0'] = self.get_site_xvelp('0_grip_site') * self.dt
        end_vel['arm1'] = self.get_site_xvelp('1_grip_site') * self.dt

        obs[0:3] = (  # gripper position in global coordinates for the first arm
            end_pos['arm0']
        )
        obs[3:6] = (  # gripper position in global coordinates for the second arm
            end_pos['arm1']
        )

        obs[6:9] = (  # gripper linear velocity for the first arm
            end_vel['arm0']
        )
        obs[9:12] = (  # gripper linear velocity for the second arm
            end_vel['arm1']
        )
        # gripper joint position and velocity for the first arm
        obs[12] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[13] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt
        # gripper joint position and velocity for the second arm
        obs[14] = self.mj_data.joint('1_r_finger_joint').qpos[0]
        obs[15] = self.mj_data.joint('1_r_finger_joint').qvel[0] * self.dt

        """Desired goal is the position of the goal site for HER algorithm."""
        desired_goal['arm0'] = self.get_site_pos('0_goal_site')
        desired_goal['arm1'] = self.get_site_pos('1_goal_site')

        # Map it into a 6-dimensional end_pos array
        end_pos = np.concatenate([end_pos['arm0'], end_pos['arm1']])
        desired_goal = np.concatenate([desired_goal['arm0'], desired_goal['arm1']])

        return {
            'observation': obs.copy(),
            'achieved_goal': end_pos.copy(),  # block position
            'desired_goal': desired_goal.copy()
        }

    def _get_info(self) -> dict:
        end_pos = {agent: 0 for agent in self.robot.agents}
        desired_goal = {agent: 0 for agent in self.robot.agents}
        """ Batch formatting for multi-agent environments."""
        end_pos['arm0'] = self.get_site_pos('0_grip_site')
        end_pos['arm1'] = self.get_site_pos('1_grip_site')
        desired_goal['arm0'] = self.get_site_pos('0_goal_site')
        desired_goal['arm1'] = self.get_site_pos('1_goal_site')
        # Map it into a 6-dimensional end_pos array
        end_pos = np.concatenate([end_pos['arm0'], end_pos['arm1']])
        desired_goal = np.concatenate([desired_goal['arm0'], desired_goal['arm1']])

        return {
            'is_success': self._is_success(end_pos, desired_goal, th=0.05)}

    def reset_object(self):
        random_goal_x_pos = np.random.uniform(0.05, 0.55)
        random_goal_y_pos = np.random.uniform(-0.15, 0.30)
        random_goal_z_pos = np.random.uniform(0.46, 0.66)
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        site_id = self.get_site_id('0_goal_site')
        self.mj_model.site_pos[site_id] = goal_pos

        random_goal_x_pos = np.random.uniform(0.25, 0.75)
        random_goal_y_pos = np.random.uniform(-0.15, 0.30)
        random_goal_z_pos = np.random.uniform(0.46, 0.66)
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        site_id = self.get_site_id('1_goal_site')
        self.mj_model.site_pos[site_id] = goal_pos


if __name__ == "__main__":
    env = DualReachPlaceEnv()
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
