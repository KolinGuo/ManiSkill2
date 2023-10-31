from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import sapien
from gym import spaces
from scipy.spatial.transform import Rotation

from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.common import clip_and_scale_action

from ..base_controller import ControllerConfig
from .pd_joint_pos import PDJointPosController


# NOTE(jigu): not necessary to inherit, just for convenience
class PDEEPosController(PDJointPosController):
    config: "PDEEPosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()

        # Pinocchio model to compute IK
        if isinstance(self.articulation, sapien.Widget):
            robot = self.articulation.robot
            self._wrapped_robot = True
        else:
            robot = self.articulation
            self._wrapped_robot = False
        self.pmodel = robot.create_pinocchio_model()
        self.qmask = np.zeros(robot.dof, dtype=bool)
        self.qmask[self.joint_indices] = 1

        if self.config.ee_link:
            self.ee_link = self.articulation.find_link_by_name(self.config.ee_link)
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.ee_link = self.joints[-1].get_child_link()
        self.ee_link_idx = self.ee_link.index
        self.ee_link = self.ee_link.entity

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.lower, 3))
        high = np.float32(np.broadcast_to(self.config.upper, 3))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.articulation.pose.inv()
        return to_base * self.ee_pose

    def reset(self):
        super().reset()
        self._target_pose = self.ee_pose_at_base

    def find_closest_equiv_qpos(self, target_qpos):
        """Find the closest equivalent qpos (revolute, +/- 2pi)"""
        for i, (joint, tgt_val, cur_val) in enumerate(
            zip(self.joints, target_qpos, self.qpos)
        ):
            if joint.type != "revolute":
                continue

            low, high = joint.limit[0]
            tgt_dist = abs(tgt_val - cur_val)
            # Check tgt_val - 2 * np.pi
            if (
                (tmp_val := tgt_val - 2*np.pi) > low and
                (tmp_dist := abs(tmp_val - cur_val)) < tgt_dist
            ):
                tgt_dist = tmp_dist
                target_qpos[i] = tmp_val
            # Check tgt_val + 2 * np.pi
            if (
                (tmp_val := tgt_val + 2*np.pi) < high and
                (tmp_dist := abs(tmp_val - cur_val)) < tgt_dist
            ):
                tgt_dist = tmp_dist
                target_qpos[i] = tmp_val
        return target_qpos

    def compute_ik(self, target_pose, max_iterations=100):
        # Assume the target pose is defined in the base frame
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            target_pose,
            initial_qpos=(self.articulation.robot.qpos if self._wrapped_robot
                          else self.articulation.qpos),
            active_qmask=self.qmask,
            max_iterations=max_iterations,
        )
        if success:
            return self.find_closest_equiv_qpos(result[self.joint_indices])
        else:
            return None

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        # Keep the current rotation and change the position
        if self.config.use_delta:
            delta_pose = sapien.Pose(action)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pose = sapien.Pose(action)

        return target_pose

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        self._target_qpos = self.compute_ik(self._target_pose)
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_pose": vectorize_pose(self._target_pose)}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            target_pose = state["target_pose"]
            self._target_pose = sapien.Pose(target_pose[:3], target_pose[3:])


@dataclass
class PDEEPosControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    frame: str = "ee"  # [base, ee]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPosController


class PDEEPoseController(PDEEPosController):
    config: "PDEEPoseControllerConfig"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(-self.config.rot_bound, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_bound, 3),
                ]
            )
        )
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = clip_and_scale_action(
            action[:3], self._action_space.low[:3], self._action_space.high[:3]
        )
        rot_action = action[3:]
        rot_norm = np.linalg.norm(rot_action)
        if rot_norm > 1:
            rot_action = rot_action / rot_norm
        rot_action = rot_action * self.config.rot_bound
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.use_delta:
            delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPoseController
