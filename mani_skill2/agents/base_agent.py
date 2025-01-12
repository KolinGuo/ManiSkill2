from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import sapien
import sapien.physx as physx
from gymnasium import spaces

from mani_skill2 import format_path
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import (
    check_urdf_config, parse_urdf_config, apply_urdf_config
)

from .base_controller import BaseController, CombinedController, ControllerConfig


@dataclass
class AgentConfig:
    """Agent configuration.

    Args:
        urdf_path: path to URDF file. Support placeholders like {PACKAGE_ASSET_DIR}.
        urdf_config: a dict to specify materials and simulation parameters when loading URDF from SAPIEN.
        controllers: a dict of controller configurations
        cameras: a dict of onboard camera configurations
    """

    urdf_path: str
    urdf_config: dict
    controllers: Dict[str, Union[ControllerConfig, Dict[str, ControllerConfig]]]
    cameras: Dict[str, CameraConfig]


class BaseAgent:
    """Base class for agents.

    Agent is an interface of the robot (physx.PhysxArticulation).

    Args:
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode: uid of controller to use
        fix_root_link: whether to fix the robot root link
        config: agent configuration
    """

    robot: Union[physx.PhysxArticulation, sapien.Widget]
    controllers: Dict[str, BaseController]

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
        config: AgentConfig = None,
    ):
        self.scene = scene
        self._control_freq = control_freq

        self.config = config or self.get_default_config()

        # URDF
        self.urdf_path = format_path(str(self.config.urdf_path))
        self.fix_root_link = fix_root_link
        self.urdf_config = self.config.urdf_config

        # Controller
        self.controller_configs = self.config.controllers
        self.supported_control_modes = list(self.controller_configs.keys())
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode

        self._load_articulation()
        self._setup_controllers()
        self.set_control_mode(control_mode)
        self._after_init()

    @classmethod
    def get_default_config(cls) -> AgentConfig:
        raise NotImplementedError

    def _load_articulation(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = self.fix_root_link

        urdf_config = parse_urdf_config(self.urdf_config, self.scene)
        check_urdf_config(urdf_config)

        # TODO(jigu): support loading multiple convex collision shapes
        apply_urdf_config(loader, urdf_config)
        self.robot: physx.PhysxArticulation = loader.load(self.urdf_path)
        assert self.robot is not None, f"Fail to load URDF from {self.urdf_path}"
        self.robot.set_name(Path(self.urdf_path).stem)

        # Cache robot link ids
        self.robot_link_ids = [link.entity.per_scene_id
                               for link in self.robot.get_links()]

    def _setup_controllers(self):
        self.controllers = OrderedDict()
        for uid, config in self.controller_configs.items():
            if isinstance(config, dict):
                self.controllers[uid] = CombinedController(
                    config, self.robot, self._control_freq
                )
            else:
                self.controllers[uid] = config.controller_cls(
                    config, self.robot, self._control_freq
                )

    def _after_init(self):
        """After initialization. E.g., caching the end-effector link."""
        pass

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    def set_control_mode(self, control_mode):
        """Set the controller and reset."""
        assert control_mode in self.supported_control_modes, \
            "{} not in supported modes: {}".format(control_mode,
                                                   self.supported_control_modes)
        self._control_mode = control_mode
        self.controller.reset()

    @property
    def controller(self):
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self):
        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.action_space

    def reset(self, init_qpos=None):
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.set_control_mode(self._default_control_mode)

    def set_action(self, action):
        if np.isnan(action).any():
            raise ValueError(f"Action cannot be NaN. Environment received {action=}")
        self.controller.set_action(action)

    def before_simulation_step(self):
        self.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_state(self) -> Dict:
        """Get current state for MPC, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        state["robot_root_pose"] = self.robot.get_root_pose()
        state["robot_root_vel"] = self.robot.get_root_velocity()
        state["robot_root_qvel"] = self.robot.get_root_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()
        state["robot_qacc"] = self.robot.get_qacc()

        # controller state
        state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])
        self.robot.set_qacc(state["robot_qacc"])

        if not ignore_controller and "controller" in state:
            self.controller.set_state(state["controller"])
