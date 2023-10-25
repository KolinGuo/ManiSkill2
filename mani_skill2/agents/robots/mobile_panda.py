import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose
from transforms3d.euler import euler2quat

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.configs.mobile_panda import defaults


class DummyMobileAgent(BaseAgent):
    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "base_pd_joint_vel_arm_pd_joint_vel"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()

        # Sanity check
        active_joints = self.robot.get_active_joints()
        assert active_joints[0].name == "root_x_axis_joint"
        assert active_joints[1].name == "root_y_axis_joint"
        assert active_joints[2].name == "root_z_rotation_joint"

        # Dummy base
        self.base_link = self.robot.get_links()[3].entity

        # Ignore collision between the adjustable body and ground
        body = self.robot.find_link_by_name("adjustable_body")
        s = body.collision_shapes[0]
        gs = s.get_collision_groups()
        gs[2] = gs[2] | 1 << 30
        s.set_collision_groups(gs)

    def get_proprioception(self):
        state_dict = super().get_proprioception()
        qpos, qvel = state_dict["qpos"], state_dict["qvel"]
        base_pos, base_orientation, arm_qpos = qpos[:2], qpos[2], qpos[3:]
        base_vel, base_ang_vel, arm_qvel = qvel[:2], qvel[2], qvel[3:]

        state_dict["qpos"] = arm_qpos
        state_dict["qvel"] = arm_qvel
        state_dict["base_pos"] = base_pos
        state_dict["base_orientation"] = base_orientation
        state_dict["base_vel"] = base_vel
        state_dict["base_ang_vel"] = base_ang_vel
        return state_dict

    @property
    def base_pose(self):
        qpos = self.robot.get_qpos()
        x, y, ori = qpos[:3]
        return Pose([x, y, 0], euler2quat(0, 0, ori))

    def set_base_pose(self, xy, ori):
        qpos = self.robot.get_qpos()
        qpos[0:2] = xy
        qpos[2] = ori
        self.robot.set_qpos(qpos)


class MobilePandaDualArm(DummyMobileAgent):
    _config: defaults.MobilePandaDualArmDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.MobilePandaDualArmDefaultConfig()

    def _after_init(self):
        super()._after_init()

        robot = self.robot
        self.rfinger1_joint = robot.find_joint_by_name("right_panda_finger_joint1")
        self.rfinger2_joint = robot.find_joint_by_name("right_panda_finger_joint2")
        self.lfinger1_joint = robot.find_joint_by_name("left_panda_finger_joint1")
        self.lfinger2_joint = robot.find_joint_by_name("left_panda_finger_joint2")

        self.rfinger1_link = robot.find_link_by_name("right_panda_leftfinger").entity
        self.rfinger2_link = robot.find_link_by_name("right_panda_rightfinger").entity
        self.lfinger1_link = robot.find_link_by_name("left_panda_leftfinger").entity
        self.lfinger2_link = robot.find_link_by_name("left_panda_rightfinger").entity

        self.rhand: sapien.Entity = robot.find_link_by_name("right_panda_hand").entity
        self.lhand: sapien.Entity = robot.find_link_by_name("left_panda_hand").entity

    def get_fingers_info(self):
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        finger_tips = [
            (self.rfinger2_joint.get_global_pose() * Pose([0, 0.035, 0])).p,
            (self.rfinger1_joint.get_global_pose() * Pose([0, -0.035, 0])).p,
            (self.lfinger2_joint.get_global_pose() * Pose([0, 0.035, 0])).p,
            (self.lfinger1_joint.get_global_pose() * Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        link_type = physx.PhysxArticulationLinkComponent
        finger_vels = [
            self.rfinger1_link.find_component_by_type(link_type).linear_velocity,
            self.rfinger2_link.find_component_by_type(link_type).linear_velocity,
            self.lfinger1_link.find_component_by_type(link_type).linear_velocity,
            self.lfinger2_link.find_component_by_type(link_type).linear_velocity,
        ]
        return np.array(finger_vels)


class MobilePandaSingleArm(DummyMobileAgent):
    _config: defaults.MobilePandaSingleArmDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.MobilePandaSingleArmDefaultConfig()

    def _after_init(self):
        super()._after_init()

        robot = self.robot
        self.finger1_joint = robot.find_joint_by_name("right_panda_finger_joint1")
        self.finger2_joint = robot.find_joint_by_name("right_panda_finger_joint2")

        self.finger1_link = robot.find_link_by_name("right_panda_leftfinger").entity
        self.finger2_link = robot.find_link_by_name("right_panda_rightfinger").entity

        self.hand: sapien.Entity = robot.find_link_by_name("right_panda_hand").entity

    def get_fingers_info(self):
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        finger_tips = [
            (self.finger2_joint.get_global_pose() * Pose([0, 0.035, 0])).p,
            (self.finger1_joint.get_global_pose() * Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        link_type = physx.PhysxArticulationLinkComponent
        finger_vels = [
            self.finger2_link.find_component_by_type(link_type).linear_velocity,
            self.finger1_link.find_component_by_type(link_type).linear_velocity,
        ]
        return np.array(finger_vels)

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                (self.finger2_joint.get_global_pose() * Pose([0, x, 0])).p,
                (self.finger1_joint.get_global_pose() * Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose(T)
