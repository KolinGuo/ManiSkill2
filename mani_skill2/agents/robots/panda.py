import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.configs.panda import defaults
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    get_entities_by_names,
    get_pairwise_contact_impulse,
)


class Panda(BaseAgent):
    _config: defaults.PandaDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.PandaDefaultConfig()

    def _after_init(self):
        self.finger1_joint, self.finger2_joint = get_entities_by_names(
            self.robot.get_joints(),
            ["panda_finger_joint1", "panda_finger_joint2"],
        )
        self.finger1_link, self.finger2_link = get_entities_by_names(
            self.robot.get_links(),
            ["panda_leftfinger", "panda_rightfinger"],
        )
        self.hand: sapien.LinkBase = get_entities_by_names(
            self.robot.get_links(), "panda_hand"
        )

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

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

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose.from_transformation_matrix(T)

    def get_fingers_info(self):
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        finger_tips = [
            self.finger2_joint.get_global_pose().transform(Pose([0, 0.035, 0])).p,
            self.finger1_joint.get_global_pose().transform(Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        finger_vels = [
            self.finger2_link.get_velocity(),
            self.finger1_link.get_velocity(),
        ]
        return np.array(finger_vels)

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.finger2_joint.get_global_pose().transform(Pose([0, x, 0])).p,
                self.finger1_joint.get_global_pose().transform(Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))


class FloatingPanda(Panda):
    _config: defaults.FloatingPandaDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.FloatingPandaDefaultConfig()
