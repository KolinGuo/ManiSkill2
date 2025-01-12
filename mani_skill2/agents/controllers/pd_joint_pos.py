from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from gymnasium import spaces

from ..base_controller import BaseController, ControllerConfig


class PDJointPosController(BaseController):
    config: "PDJointPosControllerConfig"

    def _get_joint_limits(self):
        qlimits = self.articulation.qlimit[self.joint_indices]
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(
                stiffness[i], damping[i], force_limit=force_limit[i],
                mode="acceleration"
            )
            joint.set_friction(friction[i])

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        self._start_qpos = self.qpos
        self._target_qpos = self.qpos

    def set_drive_targets(self, targets):
        qpos = self.articulation.qpos
        for i, (joint, joint_idx, target) in enumerate(
            zip(self.joints, self.joint_indices, targets)
        ):
            # If target is out of joint range for continuous joints
            if (
                ((tgt_below := target < -2 * np.pi) or target > 2 * np.pi) and
                joint.type == "revolute" and
                np.allclose(joint.limit, np.array([[-np.inf, np.inf]]))
            ):
                if tgt_below:
                    target += 2 * np.pi
                    qpos[joint_idx] += 2 * np.pi * (qpos[joint_idx] < 0)
                else:
                    target -= 2 * np.pi
                    qpos[joint_idx] -= 2 * np.pi * (qpos[joint_idx] > 0)
            joint.set_drive_target(target)
        self.articulation.set_qpos(qpos)

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic
            self._target_qpos = np.broadcast_to(action, self._start_qpos.shape)

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        self._step += 1

        # Compute the next target
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_qpos": self._target_qpos}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            self._target_qpos = state["target_qpos"]


@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDJointPosController


class PDJointPosMimicController(PDJointPosController):
    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]


class PDJointPosMimicControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosMimicController


class PDGripperController(PDJointPosController):
    def set_drive_property(self):
        self.articulation.set_gripper_pd(
            self.config.stiffness, self.config.damping, self.config.force_limit
        )
        self.articulation.left_gripper_joint.set_friction(self.config.friction)
        self.articulation.right_gripper_joint.set_friction(self.config.friction)

    def set_drive_targets(self, target: float):
        self.articulation.set_gripper_target(target)


class PDGripperControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDGripperController
