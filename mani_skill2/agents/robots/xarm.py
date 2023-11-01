from collections import OrderedDict

import numpy as np
import sapien

from mani_skill2 import format_path
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.configs.xarm import defaults
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse
from .xarm_widget import XArm7 as XArm7Widget


class XArm(BaseAgent):
    _config: defaults.XArmDefaultConfig

    @classmethod
    def get_default_config(cls):
        raise NotImplementedError()

    def _after_init(self):
        self.finger1_link: sapien.Entity = self.robot.find_link_by_name(
            "left_finger"
        ).entity
        self.finger2_link: sapien.Entity = self.robot.find_link_by_name(
            "right_finger"
        ).entity

    def check_grasp(self, entity: sapien.Entity, min_impulse=1e-6, max_angle=85):
        assert isinstance(entity, sapien.Entity), type(entity)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, entity)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, entity)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)


class XArm7(XArm):
    _config: defaults.XArm7DefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.XArm7DefaultConfig()


class XArm7D435(XArm):
    _config: defaults.XArm7D435DefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.XArm7D435DefaultConfig()

    def _load_articulation(self):
        self.robot: sapien.Widget = self.scene.load_widget(
            XArm7Widget(asset_dir=format_path("{PACKAGE_ASSET_DIR}/descriptions/"))
        )

        # Cache robot link ids
        self.robot_link_ids = [link.entity.per_scene_id for link in self.robot.links]

    def _setup_controllers(self):
        self.controllers = OrderedDict()
        for uid, config in self.controller_configs.items():
            if isinstance(config, dict):
                self.controllers[uid] = CombinedController(
                    config, self.robot, self._control_freq,
                    balance_gravity=False, balance_coriolis=True
                )
            else:
                self.controllers[uid] = config.controller_cls(
                    config, self.robot, self._control_freq
                )

    def get_proprioception(self):
        obs = super().get_proprioception()
        obs["qvel"][-1] = 0.0  # No gripper qvel
        return obs

    def reset(self, init_qpos=None):
        super().reset(init_qpos)
        # Also reset drive_target and drive_velocity_target
        if init_qpos is not None:
            self.robot.set_arm_target(init_qpos[:-1])
            self.robot.set_gripper_target(init_qpos[-1])
        self.robot.set_arm_velocity_target(np.zeros(len(self.robot.arm_joints)))


class FloatingXArm(XArm):
    _config: defaults.FloatingXArmDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.FloatingXArmDefaultConfig()


class FloatingXArmD435(XArm):
    _config: defaults.FloatingXArmD435DefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.FloatingXArmD435DefaultConfig()
