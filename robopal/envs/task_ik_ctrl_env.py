import numpy as np
from robopal.envs.jnt_imp_ctrl_env import JntCtrlEnv
import robopal.commons.transform as T


class PosCtrlEnv(JntCtrlEnv):
    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=200,
                 is_interpolate=True,
                 is_pd=False,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate
        )
        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.is_pd = is_pd
        self.vel_des = np.zeros(3)

        _, self.init_rot_quat = self.kdl_solver.fk(self.robot.single_arm.arm_qpos, rot_format='quaternion')

    @property
    def vel_cur(self):
        """ Current velocity, consist of 3*1 cartesian and 4*1 quaternion """
        J = self.kdl_solver.get_full_jac(self.robot.single_arm.arm_qpos)
        vel_cur = np.dot(J, self.robot.single_arm.arm_qvel)
        return vel_cur

    def compute_pd_increment(self, p_goal: np.ndarray,
                             p_cur: np.ndarray,
                             r_goal: np.ndarray,
                             r_cur: np.ndarray,
                             pd_goal: np.ndarray = np.zeros(3),
                             pd_cur: np.ndarray = np.zeros(3)):
        pos_incre = self.p_cart * (p_goal - p_cur) + self.d_cart * (pd_goal - pd_cur)
        quat_incre = self.p_quat * (r_goal - r_cur)
        return pos_incre, quat_incre

    def step_controller(self, action):
        if len(action) != 3 and len(action) != 7:
            raise ValueError("Invalid action length.")
        if self.is_pd is False:
            p_goal = action[:3]
            r_goal = T.quat_2_mat(self.init_rot_quat if len(action) == 3 else action[3:])
        else:
            p_cur, r_cur = self.kdl_solver.fk(self.robot.single_arm.arm_qpos, rot_format='quaternion')

            r_target = self.init_rot_quat if len(action) == 3 else action[3:]
            p_incre, r_incre = self.compute_pd_increment(p_goal=action[:3], p_cur=p_cur,
                                                         r_goal=r_target, r_cur=r_cur,
                                                         pd_goal=self.vel_des, pd_cur=self.vel_cur[:3])
            p_goal = p_incre + p_cur
            r_goal = T.quat_2_mat(r_cur + r_incre)

        return self.kdl_solver.ik(p_goal, r_goal, q_init=self.robot.single_arm.arm_qpos)

    def step(self, action):
        action = self.step_controller(action)
        return super().step(action)


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = PosCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        is_pd=True
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.33116, -0.09768533, 0.26947228, 1, 0, 0, 0])
        env.step(action)
        if env.is_render:
            env.render()