import os
import shutil
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose, look_at
from mani_skill2.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_cfgs,
)
from mani_skill2.utils.geometry import (
    get_axis_aligned_bbox_for_actor,
    angle_between_vec
)
from mani_skill2.utils.camera import resize_obs_images

from pyrl.utils.data import GDict

from .base_env import StationaryManipulationEnv
from .pick_single import build_actor_ycb


def get_axis_aligned_bbox_for_cube(cube_actor):
    assert len(cube_actor.get_collision_shapes()) == 1, "More than 1 collision"

    cube_corners = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
         [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype='float64'
    )
    cube_corners *= cube_actor.get_collision_shapes()[0].geometry.half_lengths
    mat = cube_actor.get_pose().to_transformation_matrix()
    world_corners = cube_corners @ mat[:3, :3].T + mat[:3, 3]
    mins = np.min(world_corners, 0)
    maxs = np.max(world_corners, 0)

    return mins, maxs


@register_env("PlaceCubeInBowl-v0", max_episode_steps=200)
@register_env("PlaceCubeInBowl-v1", max_episode_steps=50, extra_state_obs=True)
@register_env("PlaceCubeInBowl-v2", max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15)
@register_env("PlaceCubeInBowl-v3", max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="dense_v2",
              no_robot_static_checks=True, success_needs_ungrasp=True)
@register_env("PlaceCubeInBowl-v4", max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="dense_v2",
              no_static_checks=True, success_needs_ungrasp=True,
              check_collision_during_init=False)
@register_env("PlaceCubeInBowl-v5", max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="dense_v2",
              no_static_checks=True, success_needs_ungrasp=True,
              check_collision_during_init=False,
              robot_base_at_world_frame=True)
@register_env("PlaceCubeInBowlXArm-v5", max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="dense_v2",
              robot="xarm7", real_setup=True, image_obs_mode="sideview",
              no_static_checks=True, success_needs_ungrasp=True,
              check_collision_during_init=False)
@register_env("PlaceCubeInBowlStaged-v2",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged", stage_obs=True)
@register_env("PlaceCubeInBowlStagedNoStatic-v2",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged", stage_obs=True,
              no_static_checks=True)
@register_env("PlaceCubeInBowlStaged-v3",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v2", stage_obs=True,
              no_robot_static_checks=True)
@register_env("PlaceCubeInBowlStaged-v4",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v2", stage_obs=True,
              no_robot_static_checks=True, stage2_check_stage1=False)
#@register_env("PlaceCubeInBowlStaged-v5",
#              max_episode_steps=50, extra_state_obs=True,
#              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
#              reward_mode="sparse_staged_v2", stage_obs=True,
#              no_robot_static_checks=True, stage2_check_stage1=False,
#              no_reaching_reward_in_stage2=True)
@register_env("PlaceCubeInBowlStaged-v6",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v3", stage_obs=True,
              no_robot_static_checks=True, stage2_check_stage1=False,
              success_needs_ungrasp=True)
@register_env("PlaceCubeInBowlStaged-v7",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v3", stage_obs=True,
              no_static_checks=True, stage2_check_stage1=False,
              success_needs_ungrasp=True, check_collision_during_init=False)
@register_env("PlaceCubeInBowlStaged-v8",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v3", stage_obs=True,
              no_static_checks=True, stage2_check_stage1=False,
              success_needs_ungrasp=True, check_collision_during_init=False,
              robot_base_at_world_frame=True)
@register_env("PlaceCubeInBowlStagedXArm-v8",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="sparse_staged_v3", stage_obs=True,
              robot="xarm7", real_setup=True, image_obs_mode="sideview",
              no_static_checks=True, stage2_check_stage1=False,
              success_needs_ungrasp=True, check_collision_during_init=False)
@register_env("PlaceCubeInBowlSAMStagedXArm-v8",
              max_episode_steps=50, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              reward_mode="grounded_sam_sparse_staged_v3", stage_obs=True,
              save_trajectory=True,
              robot="xarm7", real_setup=True, image_obs_mode="sideview",
              no_static_checks=True, stage2_check_stage1=False,
              success_needs_ungrasp=True, check_collision_during_init=False)
class PlaceCubeInBowlEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    SUPPORTED_IMAGE_OBS_MODES = ("hand_base", "sideview")
    SUPPORTED_REWARD_MODES = ("dense", "dense_v2", "sparse", "sparse_staged",
                              "sparse_staged_v2", "sparse_staged_v3",
                              "grounded_sam_sparse_staged_v3")

    def __init__(self, *args,
                 asset_root: str = None,
                 model_json: str = None,
                 model_ids: List[str] = ('024_bowl'),
                 image_obs_mode=None,
                 image_obs_shape=(128, 128),
                 obj_init_rot_z=True,
                 obj_init_rot=0,
                 extra_state_obs=False,
                 fix_init_bowl_pos=False,
                 dist_cube_bowl=0.2,
                 cube_size_randomization=False,
                 bowl_size_randomization=False,
                 stage_obs=False,
                 tcp_to_cube_dist_thres=0.015,
                 check_collision_during_init=True,
                 no_static_checks=False,
                 no_robot_static_checks=False,
                 stage2_check_stage1=True,
                 no_reaching_reward_in_stage2=False,
                 success_needs_ungrasp=False,
                 ungrasp_sparse_reward=False,
                 ungrasp_reward_scale=1.0,
                 gsam_track_cfg={},
                 real_setup=False,
                 robot_base_at_world_frame=False,
                 remove_obs_extra=[],
                 save_trajectory=False,
                 **kwargs):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        # NOTE(jigu): absolute path will overwrite asset_root
        model_json = self.asset_root / format_path(model_json)

        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids

        self.model_id = model_ids[0]
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        self.original_cube_half_size = self.cube_half_size.copy()
        self.cube_size_randomization = cube_size_randomization
        self.bowl_size_randomization = bowl_size_randomization

        self.stage_obs = stage_obs
        self.num_stages = 3
        self.current_stage = np.zeros(self.num_stages).astype(bool)
        self.tcp_to_cube_dist_thres = tcp_to_cube_dist_thres
        self.extra_state_obs = extra_state_obs
        self.fix_init_bowl_pos = fix_init_bowl_pos
        self.dist_cube_bowl = dist_cube_bowl
        self.check_collision_during_init = check_collision_during_init

        # Debug success evaluation and reward
        self.no_static_checks = no_static_checks
        self.no_robot_static_checks = no_robot_static_checks
        self.stage2_check_stage1 = stage2_check_stage1
        self.no_reaching_reward_in_stage2 = no_reaching_reward_in_stage2
        self.success_needs_ungrasp = success_needs_ungrasp
        self.ungrasp_sparse_reward = ungrasp_sparse_reward
        self.ungrasp_reward_scale = ungrasp_reward_scale

        self.pmodel = None

        self.real_setup = real_setup
        self.robot_base_at_world_frame = robot_base_at_world_frame
        self.remove_obs_extra = remove_obs_extra

        self._check_assets()

        ### Grounded-SAM related ###
        self.use_grounded_sam = "grounded_sam" in kwargs.get(
            "reward_mode", self.SUPPORTED_REWARD_MODES[0]
        )
        if self.use_grounded_sam:
            self._initialize_grounded_sam(**gsam_track_cfg)
            self.recent_sam_obs = None
            self.recent_valid_sam_pts = {}
            self.sam_current_stage = np.zeros(self.num_stages).astype(bool)
        self.save_trajectory = save_trajectory
        if self.save_trajectory:
            self.episode_trajs = {}  # {traj_#: {key: value}}
            self.current_traj = defaultdict(list)
            self.episode_cnt = 0
            self.save_traj_per_episode = 20
            # Create folder to store training trajectory
            self.sam_traj_dir = Path(os.environ["log_dir"]) / 'sam_train_traj'
            if self.sam_traj_dir.is_dir():
                shutil.rmtree(self.sam_traj_dir)
            self.sam_traj_dir.mkdir(parents=True)

        # Image obs mode
        if image_obs_mode is None:
            image_obs_mode = self.SUPPORTED_IMAGE_OBS_MODES[0]
        if image_obs_mode not in self.SUPPORTED_IMAGE_OBS_MODES:
            raise NotImplementedError("Unsupported image obs mode: {}".format(image_obs_mode))
        self._image_obs_mode = image_obs_mode
        self.image_obs_shape = image_obs_shape
        if self.real_setup:
            assert self._image_obs_mode == "sideview"

        super().__init__(*args, **kwargs)

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
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
        density = self.model_db[self.model_id].get("density", 1000)
        self.bowl = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        self.bowl.name = self.model_id

    def reset(self, seed=None, reconfigure=False,
              model_id=None, model_scale=None):
        # reset stage obs
        self.current_stage = np.zeros(self.num_stages).astype(bool)
        if self.use_grounded_sam:
            self.recent_sam_obs = None
            self.recent_valid_sam_pts = {}
            self.sam_current_stage = np.zeros(self.num_stages).astype(bool)

            # Save training trajectory
            if self.save_trajectory:
                # save only if current episode has at least stepped once
                if "action" in self.current_traj:
                    self.episode_cnt += 1

                    traj_name = f"traj_{self.episode_cnt}"
                    self.episode_trajs[traj_name] = self.current_traj

                    if self.episode_cnt % self.save_traj_per_episode == 0:
                        GDict(self.episode_trajs).to_hdf5(
                            self.sam_traj_dir / f"{traj_name}.h5"
                        )
                        self.episode_trajs = {}
                self.current_traj = defaultdict(list)

        self._prev_actor_poses = {}
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure

        obs = super().reset(seed=self._episode_seed, reconfigure=reconfigure)

        if self.use_grounded_sam and self.save_trajectory:
            self.current_traj["env_states"].append(self.get_state())
            self.current_traj["sam_obs"].append(self.recent_sam_obs)
        return obs

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            if not self.bowl_size_randomization:
                model_scales = self.model_db[self.model_id].get("scales")
                if model_scales is None:
                    model_scale = 1.0
                else:
                    model_scale = random_choice(model_scales, self._episode_rng)
            else:
                model_scale = self._episode_rng.uniform(0.8, 1.2)
                reconfigure = True
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        if self.cube_size_randomization:
            self.cube_half_size = self.original_cube_half_size * self._episode_rng.uniform(0.7, 1.3)
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.bowl.set_damping(0.1, 0.1)
        self.cube = self._build_cube(self.cube_half_size,
                                     color=(0, 1, 0), name="cube")

    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_bowl_actors(self):
        # The object will fall from a certain height
        if self.real_setup:
            xy = self._episode_rng.uniform([0.4, -0.3], [0.55, 0.0], [2])
            # print(xy)
        elif self.fix_init_bowl_pos:
            xy = self._episode_rng.uniform([-0.1, -0.05], [0, 0.05], [2])
        else:
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
        self.bowl.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))
        self.cube.set_pose(Pose([10, 0, 0]))

        # Lock rotation around x and y
        self.bowl.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.bowl.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.bowl.set_pose(self.bowl.pose)
        self.bowl.set_velocity(np.zeros(3))
        self.bowl.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.bowl.velocity)
        ang_vel = np.linalg.norm(self.bowl.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

    def _initialize_actors(self, cube_ori=None):
        """cubeA_ori is the angle from bowl to A"""
        self._initialize_bowl_actors()

        if not self.real_setup:
            if cube_ori is None:
                # cube_ori = self._episode_rng.uniform(-np.pi/4, np.pi*3/4)
                cube_ori = self._episode_rng.uniform(0, 2 * np.pi)
            cube_xy = self.bowl.pose.p[:2] + \
                [np.cos(cube_ori) * self.dist_cube_bowl,
                np.sin(cube_ori) * self.dist_cube_bowl]
        else:
            dist_cube_bowl = self._episode_rng.uniform(0.15, 0.25)
            cube_ori = self._episode_rng.uniform(np.pi, 2 * np.pi)
            cube_xy = self.bowl.pose.p[:2] + \
                [np.cos(cube_ori) * dist_cube_bowl,
                np.sin(cube_ori) * dist_cube_bowl]

        cube_q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            cube_q = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.cube_half_size[2]
        cube_pose = Pose([cube_xy[0], cube_xy[1], z], cube_q)

        self.cube.set_pose(cube_pose)

    def _check_collision(self, num_steps=5) -> bool:
        for _ in range(num_steps):
            self._scene.step()

        lin_vel = np.linalg.norm(self.bowl.velocity)
        ang_vel = np.linalg.norm(self.bowl.angular_velocity)
        return lin_vel > 1e-3 or ang_vel > 1e-2

    def _initialize_agent(self):
        super()._initialize_agent()
        if self.pmodel is None:
            self.pmodel = self.agent.robot.create_pinocchio_model()
            self.ee_link_idx = self.agent.robot.get_links().index(self.tcp)

        # Set agent qpos to be at grasping cube position #
        # Build grasp pose
        cube_pose = self.cube.pose.to_transformation_matrix()
        cube_pos = cube_pose[:3, -1]
        if self.control_mode == "pd_ee_delta_pos":  # ee position control
            cur_ee_pose = self.tcp.pose
            T_world_ee_poses = [Pose(cube_pos, cur_ee_pose.q)]
        else:
            # Get the cube axis that has larger angle with cube_to_bowl
            cube_x_axis, cube_y_axis = cube_pose[:3, 0], cube_pose[:3, 1]
            cube_to_bowl = self.bowl.pose.p - cube_pos
            ang_cube_x = angle_between_vec(cube_to_bowl, cube_x_axis)
            ang_cube_x = min(ang_cube_x, np.pi - ang_cube_x)
            ang_cube_y = angle_between_vec(cube_to_bowl, cube_y_axis)
            ang_cube_y = min(ang_cube_y, np.pi - ang_cube_y)
            if ang_cube_x > ang_cube_y:
                closing = cube_x_axis
            else:
                closing = cube_y_axis

            T_world_ee_poses = [
                self.agent.build_grasp_pose([0, 0, -1], closing, cube_pos),
                self.agent.build_grasp_pose([0, 0, -1], -closing, cube_pos),
            ]
        T_world_robot = self.agent.robot.pose
        T_robot_ee_poses = [T_world_robot.inv().transform(T_we)
                            for T_we in T_world_ee_poses]

        # Compute IK
        cur_robot_qpos = self.agent.robot.get_qpos()
        for T_robot_ee in T_robot_ee_poses:
            qpos, success, error = self.pmodel.compute_inverse_kinematics(
                self.ee_link_idx, T_robot_ee,
                initial_qpos=cur_robot_qpos,
                max_iterations=100
            )

            # NOTE: Open gripper (currently, xarm7 gripper is opened at q_min)
            if 'panda' in self.robot_uid:
                open_gripper_q_idx = 1  # the "open" qlimit index of the gripper
            elif 'xarm' in self.robot_uid:
                open_gripper_q_idx = 0
            else:
                raise NotImplementedError()
            qpos[-2:] = self.agent.robot.get_qlimits()[-2:, open_gripper_q_idx]

            if (not self.check_collision_during_init) and success:
                self.robot_grasp_cube_qpos = qpos
                break
            elif self.check_collision_during_init:
                # NOTE: turn off check collision, can lead to weird placement
                if not success:  # No feasible IK
                    continue

                # Check collision
                self.agent.robot.set_qpos(qpos)  # set to target qpos
                if not self._check_collision():
                    self.robot_grasp_cube_qpos = qpos
                    self.agent.robot.set_qpos(cur_robot_qpos)  # Reset qpos
                    break
        else:
            print("[ENV] No successful grasp pose found!")
            # Attempt to reset bowl/cube position
            self._initialize_actors()
            self._initialize_agent()

    # Shift robot base frame to world frame
    def _configure_agent(self):
        super()._configure_agent()

        # Set robot base frame at world frame
        if self.robot_base_at_world_frame:
            if self.robot_uid == "panda":
                self.world_frame_delta_pos = np.array([-0.615, 0, 0], dtype=np.float32)
            else:
                raise NotImplementedError(self.robot_uid)

    def _configure_cameras(self):
        super()._configure_cameras()

        # Set robot base frame at world frame, change camera pose accordingly
        if self.robot_base_at_world_frame:
            for uid, camera_cfg in self._camera_cfgs.items():
                if uid in self._agent_camera_cfgs:
                    continue  # do not update cameras attached to agent
                else:
                    camera_cfg.p -= self.world_frame_delta_pos

    def _configure_render_cameras(self):
        super()._configure_render_cameras()

        # Set robot base frame at world frame, change camera pose accordingly
        if self.robot_base_at_world_frame:
            for uid, camera_cfg in self._render_camera_cfgs.items():
                camera_cfg.p -= self.world_frame_delta_pos

    def _initialize_task(self, max_trials=100, verbose=False):
        super()._initialize_task()

        # Shift robot and objects
        if self.robot_base_at_world_frame:
            for obj in [self.agent.robot, self.bowl, self.cube]:
                obj_pose = obj.get_pose()
                obj.set_pose(Pose(p=obj_pose.p - self.world_frame_delta_pos,
                                  q=obj_pose.q))

        bowl_pos = self.bowl.pose.p

        self.goal_pos = bowl_pos + [0, 0, 0.05]

    def _get_obs_extra(self) -> OrderedDict:
        # Update goal_pos in case the bowl moves
        self.goal_pos = self.bowl.pose.p + [0, 0, 0.05]

        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.goal_pos,
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                cube_pose=vectorize_pose(self.cube.pose),
                tcp_to_cube_pos=self.cube.pose.p - self.tcp.pose.p,
                cube_to_goal_pos=self.goal_pos - self.cube.pose.p,
            )
            if self.extra_state_obs:
                obs.update(
                    bowl_pose=vectorize_pose(self.bowl.pose),
                    tcp_to_bowl_pos=self.bowl.pose.p - self.tcp.pose.p,
                    cube_to_bowl_pos=self.bowl.pose.p - self.cube.pose.p,
                    cube_bbox=np.hstack(
                        get_axis_aligned_bbox_for_cube(self.cube)
                    ),
                    bowl_bbox=np.hstack(
                        get_axis_aligned_bbox_for_actor(self.bowl)
                    ),
                )
        if self.stage_obs:
            # NOTE: this comes one-step later than evaluate/reward
            if self.use_grounded_sam:
                obs.update(stage=self.sam_current_stage.astype(float))
            else:
                obs.update(stage=self.current_stage.astype(float))

        for obs_key in self.remove_obs_extra:
            obs.pop(obs_key, None)

        # print("goal", self.goal_pos)
        # print("bowl", self.bowl.pose)
        # print("cube", self.cube.pose)

        return obs

    def check_cube_inside(self, bowl_bbox=None, cube_bbox=None):
        """Check if the cube is placed inside the bowl"""
        if bowl_bbox is not None and cube_bbox is not None:
            bowl_mins, bowl_maxs = bowl_bbox
            cube_mins, cube_maxs = cube_bbox
        else:
            bowl_mins, bowl_maxs = get_axis_aligned_bbox_for_actor(self.bowl)
            cube_mins, cube_maxs = get_axis_aligned_bbox_for_cube(self.cube)

        # For xy axes, cube need to be inside bowl
        # For z axis, cube_z_max can be greater than bowl_z_max
        #   by its half_length
        if bowl_mins[0] <= cube_mins[0] and cube_maxs[0] <= bowl_maxs[0] and \
           bowl_mins[1] <= cube_mins[1] and cube_maxs[1] <= bowl_maxs[1] and \
           bowl_mins[2] <= cube_mins[2] and \
           cube_maxs[2] <= bowl_maxs[2] + self.cube_half_size.max():
            return True

        return False

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def check_actor_static(self, actor: sapien.Actor,
                           max_v=None, max_ang_v=None):
        """Check whether the actor is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues
        """
        from mani_skill2.utils.geometry import angle_distance

        pose = actor.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (
                np.linalg.norm(actor.get_velocity()) <= max_v
            )
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            prev_actor_pose = self._prev_actor_poses[actor.name]
            dt = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - prev_actor_pose.p) <= max_v * dt
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(prev_actor_pose, pose) <= max_ang_v * dt
            )

        # CAUTION: carefully deal with it for MPC
        self._prev_actor_poses[actor.name] = pose
        return flag_v and flag_ang_v

    def evaluate(self, **kwargs):
        is_cube_inside = self.check_cube_inside()
        is_robot_static = self.check_robot_static()
        is_cube_static = self.check_actor_static(self.cube,
                                                 max_v=0.1, max_ang_v=0.2)
        is_bowl_static = self.check_actor_static(self.bowl,
                                                 max_v=0.1, max_ang_v=0.2)
        z_axis_world = np.array([0, 0, 1])
        bowl_up_axis = self.bowl.get_pose().to_transformation_matrix()[:3, :3]\
                     @ z_axis_world
        is_bowl_upwards = abs(angle_between_vec(bowl_up_axis,
                                                z_axis_world)) < 0.1*np.pi
        is_cube_grasped = self.agent.check_grasp(self.cube)
        eval_dict = dict(
            is_cube_grasped=is_cube_grasped,
            is_cube_inside=is_cube_inside,
            is_robot_static=is_robot_static,
            is_cube_static=is_cube_static,
            is_bowl_static=is_bowl_static,
            is_bowl_upwards=is_bowl_upwards,
            success=(is_cube_inside and is_robot_static and
                     is_cube_static and is_bowl_static and is_bowl_upwards),
        )
        eval_dict.update(self.get_cost())

        if self.no_robot_static_checks:
            eval_dict["success"] = (is_cube_inside and is_cube_static and
                                    is_bowl_static and is_bowl_upwards)
        elif self.no_static_checks:
            eval_dict["success"] = is_cube_inside and is_bowl_upwards

        if self.success_needs_ungrasp:
            eval_dict["success"] = eval_dict["success"] and (not is_cube_grasped)

        if "sparse_staged" in self._reward_mode:
            self.current_stage = self.get_current_stage(eval_dict)

            eval_dict.update(dict(
                in_stage1=self.current_stage[0],
                in_stage2=self.current_stage[1],
            ))

        return eval_dict

    def get_cost(self) -> dict:
        ''' Calculate the current costs and return a dict '''
        cost = {}

        tcp_to_cube_pos = self.cube.pose.p - self.tcp.pose.p
        tcp_to_cube_dist = np.linalg.norm(tcp_to_cube_pos)
        cost["tcp_to_cube_dist"] = tcp_to_cube_dist

        cube_to_goal_dist = np.linalg.norm(self.goal_pos - self.cube.pose.p)
        cost["cube_to_goal_dist"] = cube_to_goal_dist

        return cost

    def compute_dense_reward(self, info, **kwargs):
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

    def compute_dense_reward_v2(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 10.0
            return reward

        if info["is_cube_inside"] and info["is_bowl_upwards"]:
            reward = 5.0

            # ungrasp reward
            if self.ungrasp_sparse_reward and (not info["is_cube_grasped"]):
                reward += 1.0
            elif not self.ungrasp_sparse_reward:
                max_gripper_width = self.agent.robot.get_qlimits()[-2:, -1].sum()
                gripper_width = self.agent.robot.get_qpos()[-2:].sum()
                reward += gripper_width / max_gripper_width * self.ungrasp_reward_scale
        else:
            tcp_to_cube_dist = info["tcp_to_cube_dist"]
            reaching_reward = 1 - np.tanh(5 * tcp_to_cube_dist)
            reward += reaching_reward

            if info["is_cube_grasped"]:
                reward += 1.0
                cube_to_goal_dist = info["cube_to_goal_dist"]
                place_reward = 1 - np.tanh(5 * cube_to_goal_dist)
                reward += place_reward

        return reward

    def compute_staged_reward(self, info, **kwargs) -> float:
        reward = 0.0

        if info["success"]:
            reward += self.num_stages
            return reward

        if self.current_stage[0]:
            reward += 1
        if self.current_stage[1]:
            reward += 1

        return reward

    def compute_staged_reward_v2(self, info, **kwargs) -> float:
        reward = 0.0

        if info["success"]:
            reward += self.num_stages + 1
            return reward

        if not (self.no_reaching_reward_in_stage2 and self.current_stage[1]):
            tcp_to_cube_dist = info["tcp_to_cube_dist"]
            reaching_reward = 1 - np.tanh(5 * tcp_to_cube_dist)
            reward += reaching_reward

        if self.current_stage[0]:
            reward += 1
        if self.current_stage[1]:
            reward += 1 if not self.no_reaching_reward_in_stage2 else 2

        return reward

    def compute_staged_reward_v3(
        self, info, current_stage=None, **kwargs
    ) -> float:
        reward = 0.0

        if info["success"]:
            reward = (self.num_stages - 1) * 5.0
            return reward

        if current_stage is None:
            current_stage = self.current_stage

        tcp_to_cube_dist = info["tcp_to_cube_dist"]
        if current_stage[0]:  # tcp close to cube
            reward = 1.0

        if current_stage[1]:  # is_cube_inside and is_bowl_upwards
            reward = 5.0

            # ungrasp reward
            max_gripper_width = self.agent.robot.get_qlimits()[-2:, -1].sum()
            gripper_width = self.agent.robot.get_qpos()[-2:].sum()
            reward += gripper_width / max_gripper_width
        else:
            reaching_reward = 1 - np.tanh(5 * tcp_to_cube_dist)
            reward += reaching_reward

        return reward

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        state = np.hstack([state, self.goal_pos,
                           self.current_stage.astype(float)])
        if self.use_grounded_sam:
            state = np.hstack([state, self.sam_current_stage.astype(float)])
        return state

    def set_state(self, state):
        if self.use_grounded_sam:
            self.sam_current_stage = state[-self.num_stages:].astype(bool)
            state = state[:-self.num_stages]
        self.current_stage = state[-self.num_stages:].astype(bool)
        self.goal_pos = state[-3-self.num_stages:-self.num_stages]
        super().set_state(state[:-3-self.num_stages])
        self._prev_actor_poses = {
            self.cube.name: self.cube.get_pose(),
            self.bowl.name: self.bowl.get_pose(),
        }

    ### Grounded-SAM related ###
    def _initialize_grounded_sam(
        self, aot_max_len_long_term=2,
        predict_gap=10,
        prompt_with_robot_arm=True, device="cuda",
        voxel_downsample_size=0.005, nb_neighbors=20, std_ratio=0.005,
        **kwargs
    ):
        """
        :param predict_gap: run grounded_sam per predict_gap frames
                            use mask tracking for the rest
        """
        from grounded_sam_track import GroundedSAMTrack

        self.env_object_texts = ["green cube", "red bowl"]

        self.grounded_sam_track = GroundedSAMTrack(
            aot_max_len_long_term=aot_max_len_long_term,
            predict_gap=predict_gap,
            prompt_with_robot_arm=prompt_with_robot_arm,
            device=device,
            **kwargs
        )

        # For _process_pts
        self.voxel_downsample_size = voxel_downsample_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def get_current_stage(self, info, current_stage=None):
        if current_stage is None:
            current_stage = self.current_stage

        tcp_to_cube_dist = info["tcp_to_cube_dist"]
        is_cube_inside = info["is_cube_inside"]
        is_bowl_upwards = info["is_bowl_upwards"]
        success = info["success"]

        if tcp_to_cube_dist < self.tcp_to_cube_dist_thres:
            current_stage[0] = True
        if (self.stage2_check_stage1 and
                current_stage[0] and is_cube_inside):
            current_stage[1] = True
        elif (not self.stage2_check_stage1 and
                is_cube_inside and is_bowl_upwards):
            current_stage[1] = True
        if success:
            current_stage[-1] = True
        return current_stage

    @staticmethod
    def _process_pts(
        pts_lst: Union[np.ndarray, List[np.ndarray]],
        voxel_downsample_size, nb_neighbors, std_ratio
    ) -> Union[np.ndarray, List[np.ndarray]]:
        from pyrl.utils.lib3d import np2pcd

        if isinstance(pts_lst, np.ndarray):
            pts_lst = [pts_lst]

        ret_pts_lst = []
        for pts in pts_lst:
            pcd = np2pcd(pts)
            if voxel_downsample_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_size)
            pcd_filter, inlier_inds = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
            ret_pts_lst.append(np.asarray(pcd_filter.points))

        if len(ret_pts_lst) == 1:
            return ret_pts_lst[0]

        return ret_pts_lst

    def get_obs(self) -> OrderedDict:
        """Store following observation to self.recent_sam_obs if using grounded_sam
        :return sam_rgb_images: a [n_cams, 512, 512, 3] np.uint8 np.ndarray
        :return sam_xyz_images: a [n_cams, 512, 512, 3] np.float32 np.ndarray
                                per-pixel xyz coordinates in world frame
        :return sam_xyz_masks: a [n_cams, 512, 512] np.bool np.ndarray
                               per-pixel valid mask
                               (in front of the far plane of camera frustum)
        :return pred_masks: predicted mask for sam_rgb_images,
                            a [n_cams, 512, 512] np.uint8 np.ndarray
                            mask value is index+1 in object_texts
        :return pred_phrases: a list of n_cams [n_boxes] list of pred_phrases
        :return boxes_filt: a list of n_cams[n_boxes, 4] np.ndarray
        :return object_pcds: {object_text: object_pcd}
                             object_pcd is a [n_pts, 3] np.ndarray
        :return object_filt_pcds: filtered points, {object_text: object_pcd}
                                  object_pcd is a [n_pts, 3] np.ndarray
        """
        obs = super().get_obs()

        # Store grounded_sam image observation
        if self.use_grounded_sam:
            # Render observation for grounded_sam_track
            kwargs = {}
            # Already rendered grounded_sam image views
            if (self._obs_mode == "image"
                    and self._image_obs_mode != "hand_base"):
                kwargs["camera_params_dict"] = obs["camera_param"]
                kwargs["camera_captures_dict"] = obs["image"]

            rgb_images, xyz_images, xyz_masks = self._render_rgb_pcd_images(
                **kwargs
            )
            n_cams = len(rgb_images)
            self.grounded_sam_track.ensure_num_segtracker(n_cams)

            # Images are of shape [n_cams, H, W, 3]
            # Masks are of shape [n_cams, H, W]
            rgb_images = np.stack(rgb_images, axis=0)
            xyz_images = np.stack(xyz_images, axis=0)
            xyz_masks = np.stack(xyz_masks, axis=0)
            H, W = rgb_images.shape[1:-1]

            sam_obs = OrderedDict()
            # sam_obs["sam_rgb_images"] = rgb_images
            # sam_obs["sam_xyz_images"] = xyz_images
            # sam_obs["sam_xyz_masks"] = xyz_masks

            # Run grounded_sam_track
            ret_dict = self.grounded_sam_track.predict_and_track_batch(
                rgb_images,
                [self._elapsed_steps] * n_cams,
                self.env_object_texts,
                xyz_masks,
                np.arange(n_cams)
            )
            ret_dict["pred_masks"] = np.stack(ret_dict["pred_masks"], axis=0)
            sam_obs.update(ret_dict)
            pred_masks = ret_dict["pred_masks"]  # [n_cams, H, W]

            # Extract pcd from xyz_images
            object_pcds = {}
            object_filt_pcds = {}
            for i, object_text in enumerate(self.env_object_texts):
                object_pcd = xyz_images[pred_masks == i+1]
                object_pcds[object_text] = object_pcd
                object_filt_pcds[object_text] = self._process_pts(
                    object_pcd, self.voxel_downsample_size,
                    self.nb_neighbors, self.std_ratio
                )
            sam_obs["object_pcds"] = object_pcds
            sam_obs["object_filt_pcds"] = object_filt_pcds

            self.recent_sam_obs = sam_obs

        if self._obs_mode == "image" and self._image_obs_mode != "hand_base":
            # Remove Segmentation
            for cam_name in obs["image"]:
                obs["image"][cam_name].pop("Segmentation", None)
            obs = resize_obs_images(obs, self.image_obs_shape)

        return obs

    @staticmethod
    def get_mask_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        assert gt_mask.dtype == bool, f"Got {gt_mask.dtype}"
        assert pred_mask.dtype == bool, f"Got {pred_mask.dtype}"
        assert gt_mask.shape == pred_mask.shape, \
            f"{gt_mask.shape} != {pred_mask.shape}"

        return float((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum())

    @staticmethod
    def get_bbox_iou(gt_bbox_mins, gt_bbox_maxs,
                     pred_bbox_mins, pred_bbox_maxs) -> float:
        assert np.all(gt_bbox_maxs >= gt_bbox_mins), \
            f"Diff: {gt_bbox_maxs - gt_bbox_mins}"
        assert np.all(pred_bbox_maxs >= pred_bbox_mins), \
            f"Diff: {pred_bbox_maxs - pred_bbox_mins}"

        gt_bbox_area = np.prod(gt_bbox_maxs - gt_bbox_mins)
        pred_bbox_area = np.prod(pred_bbox_maxs - pred_bbox_mins)

        inter_bbox_mins = np.maximum(gt_bbox_mins, pred_bbox_mins)
        inter_bbox_maxs = np.minimum(gt_bbox_maxs, pred_bbox_maxs)
        if np.any(inter_bbox_maxs <= inter_bbox_mins):
            return 0.0

        inter_bbox_area = np.prod(inter_bbox_maxs - inter_bbox_mins)

        return float(inter_bbox_area
            / (gt_bbox_area + pred_bbox_area - inter_bbox_area))

    def get_info(self, **kwargs) -> dict:
        info = super().get_info(**kwargs)

        if self.use_grounded_sam:
            cube_pts = self.recent_sam_obs["object_filt_pcds"][self.env_object_texts[0]]
            bowl_pts = self.recent_sam_obs["object_filt_pcds"][self.env_object_texts[1]]

            # If object is not visible, use last visible pts as current obs
            if cube_pts.size == 0:
                cube_pts = self.recent_valid_sam_pts[self.env_object_texts[0]]
                cube_visible = False
            else:
                self.recent_valid_sam_pts[self.env_object_texts[0]] = cube_pts
                cube_visible = True
            if bowl_pts.size == 0:
                bowl_pts = self.recent_valid_sam_pts[self.env_object_texts[1]]
                bowl_visible = False
            else:
                self.recent_valid_sam_pts[self.env_object_texts[1]] = bowl_pts
                bowl_visible = True

            # Extract position
            cube_pos = np.mean(cube_pts, axis=0)
            bowl_pos = np.mean(bowl_pts, axis=0)
            # Extract bbox from object_pts
            bowl_mins, bowl_maxs = bowl_pts.min(0), bowl_pts.max(0)
            cube_mins, cube_maxs = cube_pts.min(0), cube_pts.max(0)

            tcp_to_cube_dist = np.linalg.norm(cube_pos - self.tcp.pose.p)
            is_cube_inside = self.check_cube_inside(
                bowl_bbox=(bowl_mins, bowl_maxs),
                cube_bbox=(cube_mins, cube_maxs)
            )
            is_cube_grasped = bool(self.agent.robot.get_qpos()[-2:].sum() < 0.07)
            is_bowl_upwards = True  # NOTE: no checks, assume always True

            # Compute pred_mask iou
            cube_mask_iou = self.get_mask_iou(
                self.recent_sam_obs["pred_masks"] == 1,
                self._recent_gt_actor_mask == self.cube.id
            )
            bowl_mask_iou = self.get_mask_iou(
                self.recent_sam_obs["pred_masks"] == 2,
                self._recent_gt_actor_mask == self.bowl.id
            )

            # Compute position difference
            cube_pos_dist = np.linalg.norm(cube_pos - self.cube.pose.p)
            bowl_pos_dist = np.linalg.norm(bowl_pos - self.bowl.pose.p)

            # Compute pred_bbox iou
            cube_bbox_iou = self.get_bbox_iou(
                *get_axis_aligned_bbox_for_cube(self.cube),
                cube_mins, cube_maxs
            )
            bowl_bbox_iou = self.get_bbox_iou(
                *get_axis_aligned_bbox_for_actor(self.bowl),
                bowl_mins, bowl_maxs
            )

            assert self.no_static_checks, "There are still static checks"
            sam_eval_dict = dict(
                cube_visible=cube_visible,
                bowl_visible=bowl_visible,
                cube_mask_iou=cube_mask_iou,
                bowl_mask_iou=bowl_mask_iou,
                cube_pos_dist=cube_pos_dist,
                bowl_pos_dist=bowl_pos_dist,
                cube_bbox_iou=cube_bbox_iou,
                bowl_bbox_iou=bowl_bbox_iou,
                tcp_to_cube_dist=tcp_to_cube_dist,
                is_cube_inside=is_cube_inside,
                is_cube_grasped=is_cube_grasped,
                is_bowl_upwards=is_bowl_upwards,
                success=(is_cube_inside and is_bowl_upwards
                         and (not is_cube_grasped))
            )

            if "sparse_staged" in self._reward_mode:
                self.sam_current_stage = self.get_current_stage(
                    sam_eval_dict, self.sam_current_stage
                )
                sam_eval_dict.update(
                    in_stage1=self.sam_current_stage[0],
                    in_stage2=self.sam_current_stage[1],
                )

            info.update(sam_eval_dict=sam_eval_dict)

        return info

    def get_reward(self, **kwargs):
        if self.use_grounded_sam:
            return self.compute_sparse_grounded_sam_reward(**kwargs)
        elif self._reward_mode == "sparse_staged":
            return self.compute_staged_reward(**kwargs)
        elif self._reward_mode == "sparse_staged_v2":
            return self.compute_staged_reward_v2(**kwargs)
        elif self._reward_mode == "sparse_staged_v3":
            return self.compute_staged_reward_v3(**kwargs)
        elif self._reward_mode == "dense_v2":
            return self.compute_dense_reward_v2(**kwargs)
        else:
            return super().get_reward(**kwargs)

    def compute_sparse_grounded_sam_reward(self, info, **kwargs):
        if "sparse_staged_v3" in self._reward_mode:
            reward = self.compute_staged_reward_v3(info, **kwargs)

            sam_reward = self.compute_staged_reward_v3(
                info["sam_eval_dict"], self.sam_current_stage, **kwargs
            )

            info["sam/gt_reward"] = reward
            info["sam_eval_dict"]["reward"] = sam_reward

            return sam_reward
        else:
            raise NotImplementedError(self._reward_mode)

    def get_done(self, info: dict, **kwargs):
        if self.use_grounded_sam:
            return bool(info["sam_eval_dict"]["success"])
        else:
            return bool(info["success"])

    def step(self, action: Union[None, np.ndarray, Dict]):
        """When use_grounded_sam, all info keys without sam/ prefix is GT
        except for rewards during logging (rewards is sam_reward), gt_reward is GT
        """
        obs, reward, done, info = super().step(action)

        if self.use_grounded_sam:
            # Update info dict
            sam_eval_dict = info.pop("sam_eval_dict")
            for k, v in sam_eval_dict.items():
                # all sam related info are prefixed with sam/
                info[f"sam/{k}"] = v

                # Skip accuracy computation for these keys
                if ("mask_iou" in k or "pos_dist" in k or "bbox_iou" in k
                        or "visible" in k):
                    continue

                # Add accuracy eval info
                if isinstance(v, float):
                    if k == "reward":
                        info[f"sam/{k}_diff"] = v - info["sam/gt_reward"]
                    else:
                        info[f"sam/{k}_diff"] = v - info[k]
                else:  # boolean
                    info[f"sam/{k}_acc"] = (v == info[k])

        if self.use_grounded_sam and self.save_trajectory:
            self.current_traj["env_states"].append(self.get_state())
            self.current_traj["sam_obs"].append(self.recent_sam_obs)
            self.current_traj["action"].append(action)
            self.current_traj["reward"].append(reward)
            self.current_traj["done"].append(done)
            self.current_traj["info"].append(info)
        return obs, reward, done, info

    # Add multi-view cameras
    def update_render_and_take_picture_sideview(self):
        """Update render and take pictures from all cameras (non-blocking)."""
        if self._renderer_type == "client":
            # NOTE: not compatible with StereoDepthCamera
            cameras = [x.camera for x in self._render_cameras.values()]
            self._scene._update_render_and_take_pictures(cameras)
        else:
            self.update_render()
            for cam in self._render_cameras.values():
                cam.take_picture()

    def get_images_sideview(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get (raw) images from all cameras (blocking)."""
        images = OrderedDict()
        for name, cam in self._render_cameras.items():
            images[name] = cam.get_images()

        # Save gt mask for calculate SAM pred_mask iou, [n_cams, H, W]
        self._recent_gt_actor_mask = np.stack(
            [d["Segmentation"][..., 1] for d in images.values()], axis=0
        )
        return images

    def get_camera_params_sideview(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get camera parameters from all cameras."""
        params = OrderedDict()
        for name, cam in self._render_cameras.items():
            params[name] = cam.get_params()
        return params

    def _get_obs_images(self) -> OrderedDict:
        if self._image_obs_mode == "hand_base":
            return super()._get_obs_images()

        assert self._image_obs_mode == "sideview"

        self.update_render_and_take_picture_sideview()
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
            camera_param=self.get_camera_params_sideview(),
            image=self.get_images_sideview(),
        )

    def _setup_cameras(self):
        super()._setup_cameras()

        # poses = []
        # if not self.real_setup:
        #     poses.append(look_at([0.4, 0.4, 0.4], [0.0, 0.0, 0.2]))
        #     poses.append(look_at([0.4, -0.4, 0.4], [0.0, 0.0, 0.2]))
        # else:
        #     poses.append(look_at([0.4, -1.1, 0.5], [0.4, 0.2, -0.2]))

        # camera_configs = []
        # for i, pose in enumerate(poses):
        #     camera_cfg = CameraConfig(f"sideview_camera_{i}",
        #                               pose.p, pose.q, 512, 512, 1, 0.01, 10)
        #     camera_cfg.texture_names += ("Segmentation",)
        #     camera_configs.append(camera_cfg)

        # self._sideview_camera_cfgs = parse_camera_cfgs(camera_configs)

        # self._sideview_cameras = OrderedDict()
        # if self._renderer_type != "client":
        #     for uid, camera_cfg in self._sideview_camera_cfgs.items():
        #         self._sideview_cameras[uid] = Camera(
        #             camera_cfg, self._scene, self._renderer_type
        #         )

    def _register_render_cameras(self):
        camera_configs = []
        if not self.real_setup:
            pose1 = look_at([0.4, 0.4, 0.4], [0.0, 0.0, 0.2])
            pose2 = look_at([0.4, -0.4, 0.4], [0.0, 0.0, 0.2])
            camera_configs.extend([
                CameraConfig(f"render_camera_1",
                             pose1.p, pose1.q, 512, 512, 1, 0.01, 10),
                CameraConfig(f"render_camera_2",
                             pose2.p, pose2.q, 512, 512, 1, 0.01, 10),
            ])
        else:
            #pose = look_at([0.4, -1.1, 0.5], [0.4, 0.2, -0.2])
            # SAPIEN camera pose is forward(x), left(y) and up(z)
            # T @ np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
            pose = Pose([0.582913, -0.84103, 0.447668],
                        [0.663717, -0.156798, 0.153559, 0.715062])
            camera_configs.append(
                CameraConfig("render_camera",
                             pose.p, pose.q, 848, 480, np.deg2rad(43.5), 0.01, 10)
            )

        # Add Segmentation
        for camera_cfg in camera_configs:
            camera_cfg.texture_names += ("Segmentation",)

        return camera_configs

    def _clear(self):
        super()._clear()
        # self._sideview_cameras = OrderedDict()

    def _render_rgb_pcd_images(
        self, camera_params_dict=None, camera_captures_dict=None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        :return rgb_images: a list of [512, 512, 3] np.uint8 np.ndarray
        :return xyz_images: a list of [512, 512, 3] np.float32 np.ndarray
                            per-pixel xyz coordinates in world frame
        :return xyz_masks: a list of [512, 512] np.bool np.ndarray
                           per-pixel valid mask
                           (in front of the far plane of camera frustum)
        """
        if camera_params_dict is None or camera_captures_dict is None:
            self.update_render_and_take_picture_sideview()
            camera_params_dict = self.get_camera_params_sideview()
            camera_captures_dict = self.get_images_sideview()

        rgb_images = []
        xyz_images = []  # xyz_images in world coordinates
        xyz_masks = []  # xyz_mask for valid points
        for cam_name in camera_captures_dict.keys():
            camera_params = camera_params_dict[cam_name]
            camera_captures = camera_captures_dict[cam_name]

            rgba = camera_captures["Color"]
            rgb_images.append(
                np.clip(rgba[:, :, :3] * 255, 0, 255).astype(np.uint8)
            )

            pos_depth = camera_captures["Position"]
            xyz_image = pos_depth[:, :, :3]
            xyz_masks.append(pos_depth[..., 3] < 1)

            image_shape = xyz_image.shape[:2]
            T = camera_params["cam2world_gl"]
            xyz_image = xyz_image.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            xyz_images.append(xyz_image.reshape(*image_shape, 3))

        return rgb_images, xyz_images, xyz_masks


@register_env("PlaceCubeInBowlEasy-v0", max_episode_steps=20, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15)
@register_env("PlaceCubeInBowlEasy-v1", max_episode_steps=20, extra_state_obs=True,
              fix_init_bowl_pos=True, dist_cube_bowl=0.15,
              no_static_checks=True)
class PlaceCubeInBowlEasyEnv(PlaceCubeInBowlEnv):
    """Environment where robot gripper starts at grasping cube position"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_episode_steps = 20

    def _initialize_agent(self):
        super()._initialize_agent()
        self.agent.robot.set_qpos(self.robot_grasp_cube_qpos)
