import numpy as np

from robopal.envs.base import MujocoEnv
from robopal.controllers import controllers


class RobotEnv(MujocoEnv):
    """ Robot environment.

    :param robot: Robot configuration.
    :param render_mode: Choose the render mode.
    :param controller: Choose the controller.
    :param control_freq: Upper-layer control frequency. i.g. frame per second-fps
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param enable_camera_viewer: Use camera or not.
    """

    def __init__(self,
                 robot=None,
                 control_freq=200,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 camera_name=None,
                 render_mode='human',
                 ):

        super().__init__(
            robot=robot,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            camera_name=camera_name,
            render_mode=render_mode,
        )
        self.is_interpolate = is_interpolate

        # choose controller
        assert controller in controllers, f"Not supported controller, you can choose from {controllers.keys()}"
        self.controller = controllers[controller](
            self.robot,
            is_interpolate=is_interpolate,
            interpolator_config={'dof': self.robot.jnt_num, 'control_timestep': self.control_timestep}
        )

        self.kd_solver = self.controller.kd_solver  # shallow copy

        # check the control frequency
        self._n_substeps = int(self.control_timestep / self.model_timestep)
        if self._n_substeps == 0:
            raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                             "Current Model-Timestep:{}".format(self.model_timestep))

        # memorize the initial position and rotation
        self.init_pos, self.init_rot_quat = self.kd_solver.fk(self.robot.get_arm_qpos(), rot_format='quaternion')

    def auto_render(func):
        """ Automatically render the scene. """
        def wrapper(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            self.render()
            return ret

        return wrapper

    def inner_step(self, action):
        joint_inputs = self.controller.step_controller(action)
        # Send joint_inputs to simulation
        self.set_joint_ctrl(joint_inputs)

    @auto_render
    def step(self, action: np.ndarray | dict[str, np.ndarray]):
        if self.is_interpolate:
            self.controller.step_interpolator(action)
        # low-level control
        for i in range(self._n_substeps):
            super().step(action)

    def reset(self, seed=None, options=None):
        self.controller.reset()
        super().reset(seed, options)

    def gripper_ctrl(self, actuator_name: str = None, gripper_action: int = 1):
        """ Gripper control.

        :param actuator_name: Gripper actuator name.
        :param gripper_action: Gripper action, 0 for close, 1 for open.
        """
        self.mj_data.actuator(actuator_name).ctrl = -40 if gripper_action == 0 else 40

    @property
    def dt(self):
        """
        Time of each upper step in the environment.
        """
        return self._n_substeps * self.mj_model.opt.timestep
