import abc
import logging
import typing

import mujoco
import numpy as np

from robopal.commons.renderers import MjRenderer


class MujocoEnv:
    """ This environment is the base class.

    :param xml_path(str): Load xml file from xml_path to build the mujoco model.
    :param is_render(bool): Choose if use the renderer to render the scene or not.
    :param renderer(str): choose official renderer with "viewer",
    another renderer with "mujoco_viewer"
    :param control_freq(int): Upper-layer control frequency.
    Note that high frequency will cause high time-lag.
    :param enable_camera_viewer(bool): Use camera or not.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth",
            "unity",
        ],
    }

    def __init__(self,
                 robot=None,
                 control_freq=1000,
                 enable_camera_viewer=False,
                 camera_name=None,
                 render_mode='human',
                 ):

        self.robot = robot

        self.control_freq = control_freq

        self.mj_model: mujoco.MjModel = self.robot.robot_model
        self.mj_data: mujoco.MjData = self.robot.robot_data

        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0
        self.control_timestep = 0
        self.robot_dof = self.robot.jnt_num

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode in ["human", "rgb_array", "depth"]:
            self.is_render = True
        if self.render_mode is None:
            self.is_render = False
        self.renderer = MjRenderer(self.mj_model, self.mj_data, self.render_mode,
                                   enable_camera_viewer, camera_name)
        self._initialize_time()
        self._set_init_qpos()

        self._state = None

    def step(self, action):
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        if self.renderer is not None and self.renderer.render_paused:
            self.cur_time += 1
            self.inner_step(action)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            mujoco.mj_step(self.mj_model, self.mj_data)

    @abc.abstractmethod
    def inner_step(self, action):
        """  This method will be called with one-step in mujoco, before mujoco step.
        For example, you can use this method to update the robot's joint position.

        :param action: input actions
        :return: None
        """
        raise NotImplementedError

    def reset(self):
        """ Reset the simulate environment, in order to execute next episode. """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.reset_object()
        self._set_init_qpos()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.is_render:
            self.render()

    def reset_object(self):
        """ Set pose of the object. """
        pass

    def render(self, mode="human"):
        """ render mujoco """
        if self.is_render is True:
            self.renderer.render()

    def close(self):
        """ close the environment. """
        if self.renderer is not None:
            self.renderer.close()

    def _initialize_time(self):
        """ Initializes the time constants used for simulation.

        :param control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = self.mj_model.opt.timestep
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        if self.control_freq <= 0:
            raise ValueError("Control frequency {} is invalid".format(self.control_freq))
        self.control_timestep = 1.0 / self.control_freq

    def _set_init_qpos(self):
        """ Set or reset init joint position when called env reset func. """
        for j in range(len(self.robot.joint_index)):
            self.mj_data.joint(self.robot.joint_index[j]).qpos = self.robot.init_qpos[j]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_object_pose(self, obj_joint_name: str = None, obj_pose: np.ndarray = None):
        """ Set pose of the object. """
        if isinstance(obj_joint_name, str):
            assert obj_pose.shape[0] == 7
            self.mj_data.joint(obj_joint_name).qpos = obj_pose

    def set_site_pose(self, site_name: str = None, site_pos: np.ndarray = None):
        """ Set pose of the object. """
        if isinstance(site_name, str):
            site_id = self.get_site_id(site_name)
            assert site_pos.shape[0] == 3
            self.mj_model.site_pos[site_id] = site_pos

    def get_body_id(self, name: str):
        """ Get body id from body name.

        :param name: body name
        :return: body id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_body_jacp(self, name):
        """ Query the position jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacp value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """ Query the rotation jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacr value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, None, jacr, bid)
        return jacr

    def get_body_pos(self, name: str):
        """ Get body position from body name.

        :param name: body name
        :return: body position
        """
        return self.mj_data.body(name).xpos.copy()

    def get_body_quat(self, name: str):
        """ Get body quaternion from body name.

        :param name: body name
        :return: body quaternion
        """
        return self.mj_data.body(name).xquat.copy()

    def get_body_rotm(self, name: str):
        """ Get body rotation matrix from body name.

        :param name: body name
        :return: body rotation matrix
        """
        return self.mj_data.body(name).xmat.copy().reshape(3, 3)

    def get_body_xvelp(self, name: str) -> np.ndarray:
        """ Get body velocity from body name.

        :param name: body name
        :return: translational velocity of the body
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()

    def get_body_xvelr(self, name: str) -> np.ndarray:
        """ Get body rotational velocity from body name.

        :param name: body name
        :return: rotational velocity of the body
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.mj_data.qvel)
        return xvelr.copy()

    def get_site_id(self, name: str):
        """ Get site id from site name.

        :param name: site name
        :return: site id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)

    def get_site_jacp(self, name):
        """ Query the position jacobian of a mujoco site using a name string.

        :param name: The name of a mujoco site
        :return: The jacp value of the mujoco site
        """
        sid = self.get_site_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        """ Query the rotation jacobian of a mujoco site using a name string.

        :param name: The name of a mujoco site
        :return: The jacr value of the mujoco site
        """
        sid = self.get_site_id(name)
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, None, jacr, sid)
        return jacr

    def get_site_pos(self, name: str):
        """ Get body position from site name.

        :param name: site name
        :return: site position
        """
        return self.mj_data.site(name).xpos.copy()

    def get_site_xvelp(self, name: str) -> np.ndarray:
        """ Get site velocity from site name.

        :param name: site name
        :return: translational velocity of the site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()

    def get_site_xvelr(self, name: str) -> np.ndarray:
        """ Get site rotational velocity from site name.

        :param name: site name
        :return: rotational velocity of the site
        """
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.mj_data.qvel)
        return xvelr.copy()

    def get_site_quat(self, name: str):
        """ Get site quaternion from site name.

        :param name: site name
        :return: site quaternion
        """
        return self.mj_data.site(name).xquat.copy()

    def get_site_rotm(self, name: str):
        """ Get site rotation matrix from site name.

        :param name: site name
        :return: site rotation matrix
        """
        return self.mj_data.site(name).xmat.copy().reshape(3, 3)

    def get_geom_id(self, name: str | list[str]):
        """ Get geometry id from its name.

        :param name: geometry name
        :return: geometry id
        """
        if isinstance(name, str):
            return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        else:
            ids = []
            for geom in name:
                id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                ids.append(id)
            return ids

    def save_state(self):
        """ Save the state of the mujoco model. """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.mj_model, spec)
        state = np.empty(size, np.float64)
        mujoco.mj_getState(self.mj_model, self.mj_data, state, spec)
        self._state = state

    def load_state(self):
        """ Load the state of the mujoco model. """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self.mj_model, self.mj_data, self._state, spec)

    def is_contact(self, geom1: str | list[str], geom2: str | list[str], verbose=False) -> bool:
        """ Check if two geom or geom list is in contact.

        :param geom1: geom name/list
        :param geom2: geom name/list
        :return: True/False
        """
        if isinstance(geom1, str):
            geom1 = [geom1]
        if isinstance(geom2, str):
            geom2 = [geom2]

        if len(self.mj_data.contact) > 0:
            for i, geom_pair in enumerate(self.mj_data.contact.geom):
                if geom_pair[0] in geom1 and geom_pair[1] in geom2:
                    break
                if geom_pair[0] in geom2 and geom_pair[1] in geom1:
                    break
            if verbose:
                contact_info = self.mj_data.contact[i]
                name1 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact_info.geom1)
                name2 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact_info.geom2)
                dist = contact_info.dist
                logging.info("contact geom: ", name1, name2)
                print("dist: ", dist)
            return True
        return False
