import os

from robopal.robots.base import *

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class FR5Cobot(BaseRobot):
    """ FR5 robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='FR5Cobot',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="fr5_cobot",
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_tool_Link',
        )
        self.arm_joint_names = {self.agents[0]: ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6']}
        self.arm_actuator_names = {self.agents[0]: ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6']}
        self.base_link_name = {self.agents[0]: '0_base_link'}
        self.end_name = {self.agents[0]: '0_tool_Link'}

    def add_assets(self):
        goal_site_0 = """<site name="0_goal_site" pos="0.33 -0.3 0.5" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site_0)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, 0.0, 0.0, 0.0, 0.00, 0.0])}


class DualFR5Cobot(BaseRobot):
    """ Dual FR5 robots base class. """

    def __init__(self,
                 scene='dualGrasping',
                 manipulator=['FR5Cobot', 'FR5Cobot'],
                 gripper=['rethink_gripper', 'rethink_gripper'],
                 mount=['top_point_dual_left', 'top_point_dual_right'],
                 attached_body=['0_tool_Link', '1_tool_Link']
                 ):
        super().__init__(
            name="fr5_cobot",
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body=attached_body,
        )
        self.arm_joint_names = {self.agents[0]: ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6'],
                                self.agents[1]: ['1_j1', '1_j2', '1_j3', '1_j4', '1_j5', '1_j6']}
        self.arm_actuator_names = {self.agents[0]: ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6'],
                                   self.agents[1]: ['1_a1', '1_a2', '1_a3', '1_a4', '1_a5', '1_a6']}
        self.base_link_name = {self.agents[0]: '0_base_link', self.agents[1]: '1_base_link'}
        self.end_name = {self.agents[0]: '0_eef', self.agents[1]: '1_eef'}

    def add_assets(self):
        goal_site_0 = """<site name="0_goal_site" pos="0.5 0.0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />"""
        goal_site_1 = """<site name="1_goal_site" pos="0.3 0.0 0.5" size="0.02 0.02 0.02" rgba="0 1 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site_0)
        self.mjcf_generator.add_node_from_str('worldbody', goal_site_1)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([2.56082419, -0.26609859, 1.85596393, -1.58987423, -1.57080864, -0.58030651]),
                self.agents[1]: np.array([2.56082419, -0.26609859, 1.85596393, -1.58987423, -1.57080864, -0.58030651])}


class FR5Aruco(FR5Cobot):
    def __init__(self):
        super().__init__(scene='default',
                         gripper='Swab_gripper', )

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/aruco/aruco.xml')


class FR5Collide(FR5Cobot):
    def add_assets(self):
        self.mjcf_generator.add_geom(node='worldbody', name='obstacle_box', pos='0.9 0.2 0.3',
                                     size='0.4 0.05 0.2', type='box')


class FR5Grasp(FR5Cobot):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

        self.end_name = {self.agents[0]: '0_eef'}
    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([2.56082419, -0.26609859, 1.85596393, -1.58987423, -1.57080864, -0.58030651])}