from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import OrderedDict

import numpy as np
import sapien.physx as physx
from sapien import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.camera import resize_obs_images
from mani_skill2.agents.robots.xarm import FloatingXArm, FloatingXArmD435

from .base_env import GraspingEnv
from ..pick_and_place.pick_single import build_actor_ycb


@register_env("GraspWithPrompt-v0", max_episode_steps=10,
              reward_mode="dense",
              robot="floating_xarm_d435", image_obs_mode="hand")
class GraspWithPromptEnv(GraspingEnv):
    DEFAULT_ASSET_ROOTS = {
        "ycb": "{ASSET_DIR}/mani_skill2_ycb",
    }
    DEFAULT_MODEL_JSONS = {
        "ycb": "info_pick_v0.json",
    }

    SUPPORTED_OBS_MODES = ("image",)
    SUPPORTED_IMAGE_OBS_MODES = ("hand",)
    SUPPORTED_REWARD_MODES = ("dense",)
    SUPPORTED_ROBOTS = {"floating_xarm": FloatingXArm,
                        "floating_xarm_d435": FloatingXArmD435}
    agent: Union[FloatingXArm, FloatingXArmD435]

    def __init__(self, *args,
                 asset_roots: Dict[str, str] = None,
                 model_jsons: Dict[str, str] = None,
                 model_ids: Dict[str, List[str]] = {},
                 image_obs_mode=None,
                 image_obs_shape=(128, 128),
                 bg_mask_obs=False,
                 obj_init_rot_z=True,
                 obj_init_rot=0,
                 extra_state_obs=True,
                 **kwargs):
        if asset_roots is None:
            asset_roots = self.DEFAULT_ASSET_ROOTS
        self.asset_roots = {k: Path(format_path(p)) for k, p in asset_roots.items()}

        if model_jsons is None:
            model_jsons = self.DEFAULT_MODEL_JSONS
        self.model_dbs: Dict[str, Dict[str, Dict]] = {}  # {dataset: model_db}
        self.model_ids: Dict[str, List[str]] = {}  # {dataset: model_ids}
        for dataset, asset_root in self.asset_roots.items():
            model_json = asset_root / format_path(model_jsons[dataset])
            model_db = self.model_dbs[dataset] = load_json(model_json)

            model_id_lst = model_ids.get(dataset, [])
            if isinstance(model_id_lst, str):
                model_id_lst = [model_id_lst]
            if len(model_id_lst) == 0:
                model_id_lst = sorted(model_db.keys())
            assert len(model_id_lst) > 0, model_json
            self.model_ids[dataset] = model_id_lst

        self.model_datasets = list(self.model_ids.keys())
        self.model_id = (self.model_datasets[0],
                         self.model_ids[self.model_datasets[0]][0])
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.cube_half_size = np.array([0.02] * 3, np.float32)

        self.pmodel = None

        self._check_assets()

        # Image obs mode
        if image_obs_mode is None:
            image_obs_mode = self.SUPPORTED_IMAGE_OBS_MODES[0]
        if image_obs_mode not in self.SUPPORTED_IMAGE_OBS_MODES:
            raise NotImplementedError(f"Unsupported image obs mode: {image_obs_mode}")
        self._image_obs_mode = image_obs_mode
        self.image_obs_shape = image_obs_shape
        self.bg_mask_obs = bg_mask_obs

        from real_robot.envs.base_env import XArmBaseEnv
        if not isinstance(self, XArmBaseEnv):
            super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------- #
    # Configure agent and cameras
    # ---------------------------------------------------------------------- #
    def _configure_cameras(self):
        super()._configure_cameras()

        # Select camera_cfgs based on image_obs_mode
        camera_cfgs = OrderedDict()
        if self._image_obs_mode == "hand":
            camera_cfgs["hand_camera"] = self._camera_cfgs["hand_camera"]
        else:
            raise ValueError(f"Unknown image_obs_mode: {self._image_obs_mode}")

        # Add Segmentation for bg_mask_obs
        for cfg in camera_cfgs.values():
            cfg.texture_names += ("Segmentation",)
        self._camera_cfgs = camera_cfgs

    def _register_render_cameras(self):
        """Register cameras for rendering."""
        # Remove render_camera from StationaryManipulationEnv
        return []

    # ---------------------------------------------------------------------- #
    # Load model
    # ---------------------------------------------------------------------- #
    def _check_assets(self):
        for dataset, asset_root in self.asset_roots.items():
            models_dir = asset_root / "models"
            for model_id in self.model_ids[dataset]:
                model_dir = models_dir / model_id
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"{model_dir} is not found."
                        "Please download (ManiSkill2) YCB models:"
                        "`python -m mani_skill2.utils.download_asset ycb`."
                    )

                collision_file = model_dir / "collision.obj"
                if not collision_file.exists():
                    raise FileNotFoundError(
                        "convex.obj has been renamed to collision.obj. "
                        "Please re-download YCB models."
                    )

    def _load_model(self):
        dataset, model_id = self.model_id
        if dataset == "ycb":
            density = self.model_dbs[dataset][model_id].get("density", 1000)
            self.obj = build_actor_ycb(
                model_id,
                self._scene,
                scale=self.model_scale,
                density=density,
                root_dir=self.asset_roots[dataset],
            )
            self.obj.name = f"{dataset}/{model_id}"

    def _load_actors(self):
        self.ground = self._add_ground(render=self._bg_name is None)
        self._load_model()
        obj_comp = self.obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
        obj_comp.set_linear_damping(0.1)
        obj_comp.set_angular_damping(0.1)

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #
    def reset(self, seed=None, reconfigure=False, model_id=None, model_scale=None):
        self._prev_actor_poses = {}
        self._set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure

        return super().reset(seed=self._episode_seed, reconfigure=reconfigure)

    def reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function should clear the previous scene, and create a new one.
        """
        super().reconfigure()

        # Save current _scene and agent for collision checking
        self._scene_col = self._scene
        self.agent_col = self.agent
        self.obj_col = self.obj
        # Check collision between these actors during init
        self.check_col_actor_ids = (
            self.agent_col.robot_link_ids + [self.obj_col.per_scene_id]
        )
        super().reconfigure()

    def _set_model(self, model_id: Tuple[str, str], model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            dataset = random_choice(self.model_datasets, self._episode_rng)
            model_id = random_choice(self.model_ids[dataset], self._episode_rng)
            model_id = (dataset, model_id)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        model_info = self.model_dbs[self.model_id[0]][self.model_id[1]]
        if model_scale is None:
            model_scales = model_info.get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _get_init_z(self):
        bbox_min = self.model_dbs[self.model_id[0]][self.model_id[1]]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_actors(self):
        # The object will fall from a certain height
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        z = self._get_init_z()
        p = np.hstack([xy, z])
        q = [1, 0, 0, 0]

        # Rotate along z-axis
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)

        # Rotate along a random axis by a small angle
        if self.obj_init_rot > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, self.obj_init_rot)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        obj_comp = self.obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
        obj_comp.set_locked_motion_axes([0, 0, 0, 1, 1, 0])
        self._settle(0.5)

        # Unlock motion
        obj_comp.set_locked_motion_axes([0, 0, 0, 0, 0, 0])
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        obj_comp.set_linear_velocity(np.zeros(3))
        obj_comp.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(obj_comp.linear_velocity)
        ang_vel = np.linalg.norm(obj_comp.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

    # ---------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------- #
    def _get_obs_agent(self) -> OrderedDict:
        """No proprioception obs"""
        return {}

    def _get_obs_extra(self) -> OrderedDict:
        # TODO: tcp_to_prompt_pos
        return {}

    def get_obs(self) -> OrderedDict:
        """Wrapper for get_obs()"""
        obs = super().get_obs()

        if self._obs_mode == "image":
            # Remove Segmentation
            for cam_name, cam_obs in obs["image"].items():
                seg_obs = cam_obs.pop("Segmentation", None)
                if self.bg_mask_obs:
                    actor_mask = seg_obs[..., [1]]
                    cam_obs["bg_mask"] = (
                        (actor_mask == 0) | (actor_mask == self.ground.per_scene_id)
                    )
            obs = resize_obs_images(obs, self.image_obs_shape)

        return obs

    # ---------------------------------------------------------------------- #
    # Reward mode
    # ---------------------------------------------------------------------- #
    def get_reward(self, **kwargs):
        # TODO: add extra _reward_mode
        return super().get_reward(**kwargs)

    def compute_dense_reward(self, info, **kwargs):
        # TODO: finish reward
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_cube_dist = info["tcp_to_cube_dist"]
        reaching_reward = 1 - np.tanh(5 * tcp_to_cube_dist)
        reward += reaching_reward

        is_grasped = info["is_cube_grasped"]
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            cube_to_goal_dist = info["cube_to_goal_dist"]
            place_reward = 1 - np.tanh(5 * cube_to_goal_dist)
            reward += place_reward
        # static reward
        elif info["is_cube_inside"]:
            qvel = self.agent.robot.get_qvel()[:-2]
            static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
            reward += static_reward + 2

        return reward

    # ---------------------------------------------------------------------- #
    # Step
    # ---------------------------------------------------------------------- #
    def evaluate(self, **kwargs) -> dict:
        # TODO: 
        raise NotImplementedError
