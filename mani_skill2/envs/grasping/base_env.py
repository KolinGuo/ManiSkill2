from __future__ import annotations

from collections import OrderedDict
from typing import Any, SupportsFloat, Type

import cv2
import mplib
import numpy as np
import sapien
import sapien.physx as physx
from gymnasium import spaces
from sapien import Pose
from transforms3d.euler import euler2quat

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots.xarm import (
    FloatingXArm,
    FloatingXArmD435,
    XArm7,
    XArm7D435,
)
from mani_skill2.agents.utils import get_active_joint_indices
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.camera import resize_obs_images
from mani_skill2.utils.planner import get_planner, update_object_pose
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images


class GraspingEnv(BaseEnv):
    SUPPORTED_ROBOTS = {"xarm7": XArm7, "xarm7_d435": XArm7D435,
                        "floating_xarm": FloatingXArm,
                        "floating_xarm_d435": FloatingXArmD435}
    SUPPORTED_OBS_MODES = ("depth_mask", "image",)
    SUPPORTED_IMAGE_OBS_MODES = ("hand", "hand_front")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse")

    metadata: dict[str, Any] = BaseEnv.metadata | {
        "robots": SUPPORTED_ROBOTS,
        "obs_modes": SUPPORTED_OBS_MODES,
        "image_obs_modes": SUPPORTED_IMAGE_OBS_MODES,
        "reward_modes": SUPPORTED_REWARD_MODES,
    }

    agent: XArm7 | XArm7D435 | FloatingXArm | FloatingXArmD435

    def __init__(self, *args,
                 robot="xarm7_d435",
                 robot_init_qpos_noise=0.02,
                 image_obs_mode=None,
                 image_obs_shape=(128, 128),
                 gamma=0.95,  # for compute_final_reward()
                 no_agent_obs=False,
                 no_agent_qvel_obs=False,
                 no_tcp_pose_obs=False,
                 use_random_camera_pose=False,
                 random_camera_pose_per_step=False,
                 verbosity_level=1,
                 **kwargs):
        """
        :param verbosity_level: 0 is nothing, 1 is minimal, 2 is everything.
        """
        self.robot_uid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot not in self.SUPPORTED_ROBOTS:
            raise NotImplementedError(f"Unsupported robot: {robot}")

        # Image obs mode
        if image_obs_mode is None:
            image_obs_mode = self.SUPPORTED_IMAGE_OBS_MODES[0]
        if image_obs_mode not in self.SUPPORTED_IMAGE_OBS_MODES:
            raise NotImplementedError(f"Unsupported image obs mode: {image_obs_mode}")
        self._image_obs_mode = image_obs_mode
        self.image_obs_shape = image_obs_shape
        self.gamma = gamma

        self.no_agent_obs = no_agent_obs
        self.no_agent_qvel_obs = no_agent_qvel_obs
        self.no_tcp_pose_obs = no_tcp_pose_obs
        self.use_random_camera_pose = use_random_camera_pose
        self.random_camera_pose_per_step = random_camera_pose_per_step
        self.verbosity_level = verbosity_level

        # NOTE: sapien.PinocchioModel.compute_inverse_kinematics() does not check qlimit
        # switch to mplib.Planner
        # self.pmodel: sapien.PinocchioModel = None  # for _check_feasible_grasp_pose
        self.planner: mplib.Planner = None  # for IK and collision checking

        # NOTE: this goal position is feasible for any TCP orientation
        self.tcp_goal_pos = [0.25, -0.25, 0.35]

        super().__init__(*args, **kwargs)

        # Update action_space
        self.action_space = spaces.Tuple([
            self.agent.action_space, spaces.Discrete(2)  # {0, 1}
        ])

    # ---------------------------------------------------------------------- #
    # Load object models
    # ---------------------------------------------------------------------- #
    def _add_table_as_ground(self):
        """Add table top surface as ground plane
        https://github.com/haosulab/SAPIEN/blob/280309d3604f67e99e5f1d6645b77a53432f5c83/python/py_package/wrapper/scene.py#L115
        """
        table_bounds = np.array([[-0.095, 0.668], [-0.494, 0.075]])
        table_center = table_bounds.mean(1)
        table_half_length = (table_bounds[:, 1] - table_bounds[:, 0]) / 2

        render_material = self._renderer.create_material()
        render_material.base_color = [0.06, 0.08, 0.12, 1]
        render_material.metallic = 0.0
        render_material.roughness = 0.9
        render_material.specular = 0.8

        builder = self._scene.create_actor_builder()
        builder.add_box_visual(
            Pose(p=[*table_center, -0.05], q=[1, 0, 0, 0]),
            half_size=[*table_half_length, 0.05],
            material=render_material,
            name="",
        )
        builder.add_box_collision(
            Pose(p=[*table_center, -0.05], q=[1, 0, 0, 0]),
            half_size=[*table_half_length, 0.05],
            material=None,
        )
        builder.set_physx_body_type("static")
        ground = builder.build()
        ground.name = "ground"
        return ground

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.render.RenderMaterial = None,
    ) -> sapien.Entity:
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.base_color = (*color, 1.0)

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_turntable(
        self,
        radius: float = 0.144,  # 28.8 cm diameter
        half_length: float = 0.026,  # 5.2 cm height
        color=(0.2, 0.2, 0.2),
        name="turntable"
    ) -> sapien.Entity:
        """Build a turntable as a cylinder"""
        builder = self._scene.create_actor_builder()
        builder.add_cylinder_collision(radius=radius, half_length=half_length)
        builder.add_cylinder_visual(radius=radius, half_length=half_length,
                                    material=color)
        return builder.build_kinematic(name)

    # ---------------------------------------------------------------------- #
    # Initialize object actors
    # ---------------------------------------------------------------------- #
    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_turntable_actor(self, rot_speed=None):
        """Initialize turntable actor
        :param rot_speed: rotation speed in rad/second. CCW if positive, CW if negative.
        """
        comp = self.turntable.find_component_by_type(physx.PhysxRigidDynamicComponent)
        cylinder_col = comp.collision_shapes[0]
        radius = cylinder_col.radius
        half_length = cylinder_col.half_length
        x = self._episode_rng.uniform(-0.095 + radius, 0.668 - radius)
        y = self._episode_rng.uniform(-0.494 + radius, 0.075 - radius)
        p = [x, y, half_length]
        q = euler2quat(0, -np.pi / 2, 0)  # x-axis points up
        self.turntable.set_pose(Pose(p, q))

        if rot_speed is None:
            # By default, rotates clockwise
            rot_speed = -2 * np.pi / self._episode_rng.uniform(18.3, 49)

        self.turntable_delta_pose = Pose(
            q=euler2quat(rot_speed * self.sim_timestep, 0.0, 0.0)
        )

    # ---------------------------------------------------------------------- #
    # Configure cameras
    # ---------------------------------------------------------------------- #
    @property
    def image_obs_mode(self):
        return self._image_obs_mode

    def _register_cameras(self):
        """Register (non-agent) cameras for environment observation."""
        camera_configs = super()._register_cameras()  # base_camera

        # front_camera
        # SAPIEN camera pose is forward(x), left(y) and up(z)
        # T @ np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        from real_robot.sensors.camera import CALIB_CAMERA_POSES
        pose = CALIB_CAMERA_POSES["front_camera"]
        camera_configs.append(
            CameraConfig("front_camera", pose.p, pose.q, 848, 480,
                         np.deg2rad(43.5), 0.01, 10)
        )
        return camera_configs

    def _configure_cameras(self):
        super()._configure_cameras()

        # Select camera_cfgs based on image_obs_mode
        camera_cfgs = OrderedDict()
        render_camera_cfgs = OrderedDict()
        if self._image_obs_mode == "hand":
            camera_cfgs["hand_camera"] = self._camera_cfgs["hand_camera"]
            render_camera_cfgs["front_camera"] = self._camera_cfgs["front_camera"]
        elif self._image_obs_mode == "hand_front":
            camera_cfgs["front_camera"] = self._camera_cfgs["front_camera"]
            camera_cfgs["hand_camera"] = self._camera_cfgs["hand_camera"]
        else:
            raise ValueError(f"Unknown image_obs_mode: {self._image_obs_mode}")

        # Add segmentation masks
        for cfg in camera_cfgs.values():
            cfg.texture_names += ("Segmentation",)
        self._camera_cfgs = camera_cfgs
        self._render_camera_cfgs = render_camera_cfgs

    def _register_render_cameras(self):
        """Register cameras for rendering."""
        return self._render_camera_cfgs

    def _configure_render_cameras(self):
        super()._configure_render_cameras()

    def _setup_cameras(self):
        """Setup cameras in the scene. Called by `self.reconfigure`"""
        super()._setup_cameras()

        # Update camera intrinsics
        if "hand_camera" in self._cameras:
            K_rgb = np.array([
                [605.12158203125, 0.0, 424.5927734375],
                [0.0, 604.905517578125, 236.668975830078],
                [0.0, 0.0, 1.0],
            ])
            self._cameras["hand_camera"].camera.set_perspective_parameters(
                0.01, 100.0, K_rgb[0, 0], K_rgb[1, 1],
                K_rgb[0, 2], K_rgb[1, 2], K_rgb[0, 1]
            )

    def _randomize_camera_pose(self):
        for cam_name, camera in self._cameras.items():
            if cam_name == "hand_camera":
                delta_pose = Pose(
                    p=self._episode_rng.uniform([-0.005, -0.02, -0.005],
                                                [0.02, 0.005, 0.01]),
                    q=euler2quat(*np.deg2rad(self._episode_rng.uniform([-2, -5, -5],
                                                                       [2, 5, 5])))
                )
            else:  # larger randomization
                delta_pose = Pose(
                    p=self._episode_rng.uniform(-0.1, 0.1, size=3),
                    q=euler2quat(*np.deg2rad(self._episode_rng.uniform(-10, 10,
                                                                       size=3)))
                )
            camera.camera.local_pose = camera.camera_cfg.pose * delta_pose

    # ---------------------------------------------------------------------- #
    # Load and configure agent
    # ---------------------------------------------------------------------- #
    def _configure_agent(self):
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()

    def _load_agent(self):
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Entity = self.agent.robot.find_link_by_name(
            self.agent.config.ee_link_name
        ).entity

    def _initialize_agent(self):
        if self.robot_uid == 'xarm7':
            qpos = np.array(
                [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2,
                 0.0453556139430441, 0.0453556139430441]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([0.0, 0.0, 0.0]))
        elif self.robot_uid == 'xarm7_d435':
            qpos = self.agent.robot.init_qpos.copy()
            qpos[:-1] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 1
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([0.0, 0.0, 0.0]))
        elif self.robot_uid in ['floating_xarm', 'floating_xarm_d435']:
            # fmt: off
            qpos = np.array(
                [0.0, 0.0, 0.35886714, 0.0, 0.0, -np.pi,
                 0.0453556139430441, 0.0453556139430441]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([0.0, 0.0, 0.0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def initialize_episode(self):
        # Detach cube
        self.planner.planning_world.detach_object("cube")

        super().initialize_episode()

        if self.verbosity_level >= 1:
            print(f"[ENV] Finish initializing episode (seed={self._episode_seed})")

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #
    def reset(
        self, *, seed: int | None = None, reconfigure: bool = False, **kwargs
    ) -> tuple[dict, dict[str, Any]]:
        self._set_episode_rng(seed)

        return super().reset(seed=self._episode_seed, reconfigure=reconfigure, **kwargs)

    def _set_episode_rng(self, seed: int):
        """Set the random generator for current episode."""
        super()._set_episode_rng(seed)
        # Seed mplib as well
        mplib.set_global_seed(self._episode_seed)

    # ---------------------------------------------------------------------- #
    # Helpful functions
    # ---------------------------------------------------------------------- #
    def reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function should clear the previous scene, and create a new one.
        """
        super().reconfigure()

        # Creates planner with all articulations/actors in the environment
        if isinstance(self.agent.robot, sapien.Widget):
            robot = self.agent.robot.robot
            self._wrapped_robot = True
        else:
            robot = self.agent.robot
            self._wrapped_robot = False
        self.planner = get_planner(self, move_group="link_tcp")
        self.planner.acm.set_entry("link_base", "ground", True)
        self.qmask = np.ones(robot.dof, dtype=bool)  # mask to disable joints
        self.qmask[self.planner.move_group_joint_indices] = False

    def _check_feasible_grasp_pose(self, tcp_pos: np.ndarray) -> bool:
        """Build a grasp pose at tcp_pos, keeping gripper orientation.
        Check if ee_pose is feasible and collision free using IK"""
        # Update planner objects pose with current environment state
        update_object_pose(self.planner, self)

        pose_world_ee = Pose(tcp_pos, self.tcp.pose.q)
        pose_robot_ee = self.agent.robot.pose.inv() * pose_world_ee

        cur_robot_qpos = (self.agent.robot.robot.qpos if self._wrapped_robot
                          else self.agent.robot.qpos)

        # feasible collision-free IK
        ik_status, q_goals = self.planner.IK(
            np.hstack((pose_robot_ee.p, pose_robot_ee.q)),
            start_qpos=cur_robot_qpos,
            mask=self.qmask,
            n_init_qpos=1,  # to match previous pmodel.compute_inverse_kinematics()
            verbose=self.verbosity_level >= 2,  # print collision info if any exists
        )

        if ik_status == "Success":
            self.robot_grasp_qpos = q_goals[0][:self.agent.robot.dof]
            return True
        return False

    def check_robot_static(self, thresh=0.2):
        # Assume that the last DoF is gripper
        qvel = self.agent.robot.qvel[:-1]
        return np.max(np.abs(qvel)) <= thresh

    # ---------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------- #
    def get_obs(self) -> OrderedDict:
        """Wrapper for get_obs()"""
        obs = super().get_obs()

        if self._obs_mode == "image":
            obs = resize_obs_images(obs, self.image_obs_shape)
        return obs

    def _get_obs_agent(self) -> OrderedDict:
        if self.no_agent_obs:
            return OrderedDict()

        obs = self.agent.get_proprioception()
        if self.no_agent_qvel_obs:
            obs.pop("qvel", None)
        return obs

    def _get_obs_extra(self) -> OrderedDict:
        if self.no_tcp_pose_obs:
            return OrderedDict()

        return OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def step(
        self, action: None | np.ndarray | tuple[np.ndarray, int]
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Last dimension of action is a discrete finish indicator (1 is finished)"""
        self._elapsed_steps += 1

        if self.use_random_camera_pose and self.random_camera_pose_per_step:
            self._randomize_camera_pose()

        agent_action, finish_action = None, 0
        if action is not None:
            agent_action, finish_action = action
            assert isinstance(finish_action, int), \
                f"finish_action must be int, got {finish_action=}"

        if finish_action == 1:
            obs, reward, terminated, info = self.execute_ending_action()
        else:
            self.step_action(agent_action)
            obs = self.get_obs()
            info = self.get_info(obs=obs)
            reward = self.get_reward(obs=obs, action=action, info=info)
            terminated = bool(info["success"])

        info["finish_action"] = finish_action
        return obs, reward, terminated, False, info

    def execute_ending_action(self) -> tuple[dict, SupportsFloat, bool, dict[str, Any]]:
        """Performs ending manipulation trajectory after policy outputs finish signal
        If object is not grasped at the start of this function, no ending action
            will be executed. Reward is from compute_normalized_dense_reward()
        Otherwise, reward is from compute_normalized_final_reward()
        Outputs obs, reward, True, info
        """
        raise NotImplementedError("Need to implement execute_ending_action")

    def compute_final_reward(self, obs, info) -> float:
        """Computes the final reward after performing ending manipulation trajectory"""
        raise NotImplementedError("Need to implement compute_final_reward")

    def compute_normalized_final_reward(self, **kwargs) -> float:
        raise NotImplementedError("Need to implement compute_normalized_final_reward")

    def step_action(self, action: None | np.ndarray | tuple[np.ndarray, int]) -> None:
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray):
            self.agent.set_action(action)
        else:
            raise TypeError(type(action))

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.agent.before_simulation_step()
            self._scene.step()
            self._after_simulation_step()

    def _before_simulation_step(self):
        """For simulating dynamic scene, called inside `step_action()`"""
        pass

    def _update_turntable_actor(self):
        """Change turntable pose before each simulation step"""
        self.turntable_comp.kinematic_target = (
            self.turntable.pose * self.turntable_delta_pose
        )

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    # -------------------------------------------------------------------------- #
    # Simulation state (for restoring environment)
    # -------------------------------------------------------------------------- #
    def get_contacts_state(self) -> list[dict[str, list]]:
        """Get environment contacts state"""
        contacts_state = []
        for contact in self._scene.get_contacts():
            contacts_state.append({
                "entity_ids": [c.entity.per_scene_id for c in contact.components],
                "points": [
                    dict(
                        position=point.position,
                        impulse=point.impulse,
                        normal=point.normal,
                        separation=point.separation,
                    )
                    for point in contact.points
                ],
            })
        return contacts_state

    def get_state(self) -> dict:
        """Get environment state. Override to include task information (e.g., goal)"""
        return dict(sim=self.get_sim_state(), contacts=self.get_contacts_state())

    def set_state(self, state: dict | np.ndarray):
        """Set environment state. Override to include task information (e.g., goal)"""
        if isinstance(state, dict):
            return self.set_sim_state(state["sim"])
        else:
            return self.set_sim_state(state)

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    def render_cameras(self) -> np.ndarray | None:
        """Render function when render_mode='cameras'"""
        images = []
        if len(self._render_cameras) > 0:
            images = [
                cv2.resize(
                    self.render_rgb_array(),
                    (384, 384),  # (W, H)
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
            ]
        self.update_render()
        self.take_picture()
        cameras_images = self.get_obs()["image"]
        for camera_images in cameras_images.values():
            images.extend(observations_to_images(camera_images))
        if len(images) == 0:
            return None
        return tile_images(images)
