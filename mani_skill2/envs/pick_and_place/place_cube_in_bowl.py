import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

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
class PlaceCubeInBowlEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    SUPPORTED_IMAGE_OBS_MODES = ("hand_base", "sideview")
    SUPPORTED_REWARD_MODES = ("dense", "dense_v2", "sparse", "sparse_staged",
                              "sparse_staged_v2", "sparse_staged_v3",
                              "sparse_last_grounded_sam")

    def __init__(self, *args,
                 asset_root: str = None,
                 model_json: str = None,
                 model_ids: List[str] = ('024_bowl'),
                 image_obs_mode=None,
                 obj_init_rot_z=True,
                 obj_init_rot=0,
                 extra_state_obs=False,
                 fix_init_bowl_pos=False,
                 dist_cube_bowl=0.2,
                 stage_obs=False,
                 tcp_to_cube_dist_thres=0.015,
                 no_static_checks=False,
                 no_robot_static_checks=False,
                 stage2_check_stage1=True,
                 no_reaching_reward_in_stage2=False,
                 success_needs_ungrasp=False,
                 ungrasp_sparse_reward=False,
                 ungrasp_reward_scale=1.0,
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

        self.stage_obs = stage_obs
        self.num_stages = 3
        self.current_stage = np.zeros(self.num_stages).astype(bool)
        self.tcp_to_cube_dist_thres = tcp_to_cube_dist_thres
        self.extra_state_obs = extra_state_obs
        self.fix_init_bowl_pos = fix_init_bowl_pos
        self.dist_cube_bowl = dist_cube_bowl

        # Debug success evaluation and reward
        self.no_static_checks = no_static_checks
        self.no_robot_static_checks = no_robot_static_checks
        self.stage2_check_stage1 = stage2_check_stage1
        self.no_reaching_reward_in_stage2 = no_reaching_reward_in_stage2
        self.success_needs_ungrasp = success_needs_ungrasp
        self.ungrasp_sparse_reward = ungrasp_sparse_reward
        self.ungrasp_reward_scale = ungrasp_reward_scale

        self.pmodel = None

        self.grounded_sam = None

        self._check_assets()
        super().__init__(*args, **kwargs)

        # Image obs mode
        if image_obs_mode is None:
            image_obs_mode = self.SUPPORTED_IMAGE_OBS_MODES[0]
        if image_obs_mode not in self.SUPPORTED_IMAGE_OBS_MODES:
            raise NotImplementedError("Unsupported image obs mode: {}".format(image_obs_mode))
        self._image_obs_mode = image_obs_mode

        self.max_episode_steps = 50

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
        self._prev_actor_poses = {}
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure

        return super().reset(seed=self._episode_seed, reconfigure=reconfigure)

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
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
        if self.fix_init_bowl_pos:
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

        if cube_ori is None:
            # cube_ori = self._episode_rng.uniform(-np.pi/4, np.pi*3/4)
            cube_ori = self._episode_rng.uniform(0, 2 * np.pi)
        cube_xy = self.bowl.pose.p[:2] + \
            [np.cos(cube_ori) * self.dist_cube_bowl,
             np.sin(cube_ori) * self.dist_cube_bowl]

        cube_q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            cube_q = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.cube_half_size[2]
        cube_pose = Pose([cube_xy[0], cube_xy[1], z], cube_q)

        self.cube.set_pose(cube_pose)

    def _check_collision(self, num_steps=1) -> bool:
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
            if not success:  # No feasible IK
                continue

            # Check collision
            self.agent.robot.set_qpos(qpos)  # set to target qpos
            if not self._check_collision():
                self.robot_grasp_cube_qpos = qpos
                self.agent.robot.set_qpos(cur_robot_qpos)  # Reset qpos
                break
        else:
            print("[ENV] No successful collision-free grasp pose found!")
            # Attempt to reset bowl/cube position
            self._initialize_actors()
            self._initialize_agent()

    def _initialize_task(self, max_trials=100, verbose=False):
        bowl_pos = self.bowl.pose.p

        self.goal_pos = bowl_pos + [0, 0, 0.05]

        if self._reward_mode == "sparse_last_grounded_sam":
            self._initialize_grounded_sam()

    def _get_obs_extra(self) -> OrderedDict:
        # Update goal_pos in case the bowl moves
        self.goal_pos = self.bowl.pose.p + [0, 0, 0.05]

        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
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
                obs.update(
                    stage=self.current_stage.astype(float)
                )
        return obs

    def check_cube_inside(self):
        # Check if the cube is placed inside the bowl
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

        if self._reward_mode in ["sparse_staged", "sparse_staged_v2", "sparse_staged_v3"]:
            tcp_to_cube_dist = eval_dict["tcp_to_cube_dist"]
            if tcp_to_cube_dist < self.tcp_to_cube_dist_thres:
                self.current_stage[0] = True
            if (self.stage2_check_stage1 and
                    self.current_stage[0] and is_cube_inside):
                self.current_stage[1] = True
            elif (not self.stage2_check_stage1 and
                    is_cube_inside and is_bowl_upwards):
                self.current_stage[1] = True
            if eval_dict["success"]:
                self.current_stage[-1] = True

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
            else:
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

    def compute_staged_reward_v3(self, info, **kwargs) -> float:
        reward = 0.0

        if info["success"]:
            reward = (self.num_stages - 1) * 5.0
            return reward

        tcp_to_cube_dist = info["tcp_to_cube_dist"]
        if self.current_stage[0]:  # tcp close to cube
            reward = 1.0

        if self.current_stage[1]:  # is_cube_inside and is_bowl_upwards
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
        return np.hstack([state, self.goal_pos,
                          self.current_stage.astype(float)])

    def set_state(self, state):
        self.current_stage = state[-self.num_stages:].astype(bool)
        self.goal_pos = state[-3-self.num_stages:-self.num_stages]
        super().set_state(state[:-3-self.num_stages])
        self._prev_actor_poses = {
            self.cube.name: self.cube.get_pose(),
            self.bowl.name: self.bowl.get_pose(),
        }

    # Grounded-SAM related
    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse_last_grounded_sam":
            if self._elapsed_steps >= self.max_episode_steps:  # Last step
                return self.compute_sparse_grounded_sam_reward(**kwargs)
            else:
                return 0.0
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

    # Add multi-view cameras
    def take_picture_sideview(self):
        """Take pictures from all cameras (non-blocking)."""
        for cam in self._sideview_cameras.values():
            cam.take_picture()

    def get_images_sideview(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get (raw) images from all cameras (blocking)."""
        images = OrderedDict()
        for name, cam in self._sideview_cameras.items():
            images[name] = cam.get_images()
        return images

    def get_camera_params_sideview(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get camera parameters from all cameras."""
        params = OrderedDict()
        for name, cam in self._sideview_cameras.items():
            params[name] = cam.get_params()
        return params

    def _get_obs_images(self) -> OrderedDict:
        if self._image_obs_mode == "hand_base":
            return super()._get_obs_images()

        assert self._image_obs_mode == "sideview"

        if self._renderer_type == "client":
            # NOTE: not compatible with StereoDepthCamera
            cameras = [x.camera for x in self._sideview_cameras.values()]
            self._scene._update_render_and_take_pictures(cameras)
        else:
            self.update_render()
            self.take_picture_sideview()
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
            camera_param=self.get_camera_params_sideview(),
            image=self.get_images_sideview(),
        )

    def _setup_cameras(self):
        super()._setup_cameras()

        poses = []
        poses.append(look_at([0.4, 0.4, 0.4], [0.0, 0.0, 0.2]))
        poses.append(look_at([0.4, -0.4, 0.4], [0.0, 0.0, 0.2]))

        camera_configs = []
        for i, pose in enumerate(poses):
            camera_cfg = CameraConfig(f"sideview_camera_{i}",
                                      pose.p, pose.q, 512, 512, 1, 0.01, 10)
            camera_cfg.texture_names += ("Segmentation",)
            camera_configs.append(camera_cfg)

        self._sideview_camera_cfgs = parse_camera_cfgs(camera_configs)

        self._sideview_cameras = OrderedDict()
        if self._renderer_type != "client":
            for uid, camera_cfg in self._sideview_camera_cfgs.items():
                self._sideview_cameras[uid] = Camera(
                    camera_cfg, self._scene, self._renderer_type
                )

    def _clear(self):
        super()._clear()
        self._sideview_cameras = OrderedDict()

    def render_rgb_pcd_images(self):
        """
        :return rgb_images: a list of [512, 512, 3] np.uint8 np.ndarray
        :return xyz_images: a list of [512, 512, 3] np.float32 np.ndarray
                            per-pixel xyz coordinates in world frame
        :return xyz_masks: a list of [512, 512] np.bool np.ndarray
                           per-pixel valid mask
                           (in front of the far plane of camera frustum)
        """
        self.update_render()

        rgb_images = []
        xyz_images = []  # xyz_images in world coordinates
        xyz_masks = []   # xyz_mask for valid points
        for camera in self._sideview_cameras.values():
            camera_captures = camera.get_images(take_picture=True)

            rgba = camera_captures["Color"]
            rgb_images.append(
                np.clip(rgba[:, :, :3] * 255, 0, 255).astype(np.uint8)
            )

            pos_depth = camera_captures["Position"]
            xyz_image = pos_depth[:, :, :3]
            xyz_masks.append(pos_depth[..., 3] < 1)

            image_shape = xyz_image.shape[:2]
            T = camera.camera.get_model_matrix()
            xyz_image = xyz_image.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            xyz_images.append(xyz_image.reshape(*image_shape, 3))

        return rgb_images, xyz_images, xyz_masks

    def _initialize_grounded_sam(self):
        if self.grounded_sam is None:
            from grounded_sam import GroundedSAM
            self.grounded_sam = GroundedSAM(
                grounding_dino_model_variant='swin-b',
                sam_model_variant='vit_h',
                device="cuda:0"
            )

            import os
            self.grounded_sam_output_dir = Path(os.environ["log_dir"]) \
                / 'grounded_sam'

    def compute_sparse_grounded_sam_reward(self, info, **kwargs) -> float:
        # Render RGB images and pointclouds
        rgb_images, xyz_images, xyz_masks = self.render_rgb_pcd_images()

        self.objects_text = ["green cube", "red bowl"]
        self.prompt_with_robot_arm = True

        def find_most_confident_label(pred_phrases, label):
            pred_label_idx = None
            best_logit = 0.0
            for i, pred_phrase in enumerate(pred_phrases):
                pred_label, pred_logit = re.split('\(|\)', pred_phrase)[:2]
                if label.lower() in pred_label and float(pred_logit) > best_logit:
                    pred_label_idx = i
                    best_logit = float(pred_logit)
            return pred_label_idx

        def save_pred_result(correct_pred: bool, rgb_images, text_prompt,
                             boxes_filt_batch: List[np.ndarray],
                             pred_phrases_batch: List[List[str]],
                             pred_masks_batch: List[np.ndarray]):
            for img_i, rgb_image in enumerate(rgb_images):
                boxes_filt = boxes_filt_batch[img_i]
                pred_phrases = pred_phrases_batch[img_i]
                pred_masks = pred_masks_batch[img_i]
                # Save pred_mask results
                import uuid
                self.grounded_sam.save_pred_result(
                    self.grounded_sam_output_dir / (f'{correct_pred}_{self._elapsed_steps}_'+str(uuid.uuid4())),
                    rgb_image, text_prompt,
                    boxes_filt, pred_phrases, pred_masks
                )

        text_prompt = '. '.join(self.objects_text) + '.'
        if self.prompt_with_robot_arm:
            text_prompt += ' robot arm.'

        boxes_filt_batch, pred_phrases_batch, pred_masks_batch, _ = self.grounded_sam.predict(
            np.stack(rgb_images), [text_prompt]*len(rgb_images)
        )

        # Convert torch.Tensor to np.ndarray
        boxes_filt_batch = [b.numpy() for b in boxes_filt_batch]
        pred_masks_batch = [m.cpu().numpy() for m in pred_masks_batch]

        object_pcds = {}  # {object_text: object_pcd}
        for img_i, rgb_image in enumerate(rgb_images):
            xyz_image = xyz_images[img_i]
            xyz_mask = xyz_masks[img_i]
            boxes_filt = boxes_filt_batch[img_i]
            pred_phrases = pred_phrases_batch[img_i]
            pred_masks = pred_masks_batch[img_i]

            for object_text in self.objects_text:
                pred_label_idx = find_most_confident_label(pred_phrases, object_text)
                if pred_label_idx is None:  # object cannot be found
                    save_pred_result(
                        info["success"] == False, rgb_images,
                        text_prompt + f'(env {info["success"]}, pred {False})',
                        boxes_filt_batch, pred_phrases_batch, pred_masks_batch
                    )
                    return 0.0
                pred_mask = pred_masks[pred_label_idx, 0]  # [H, W]

                object_pcds[object_text] = xyz_image[xyz_mask & pred_mask]

        # Extract bbox from object_pcds
        bowl_mins = object_pcds["red bowl"].min(0)
        bowl_maxs = object_pcds["red bowl"].max(0)
        cube_mins = object_pcds["green cube"].min(0)
        cube_maxs = object_pcds["green cube"].max(0)

        # For xy axes, cube need to be inside bowl
        # For z axis, cube_z_max can be greater than bowl_z_max
        #   by its half_length
        if bowl_mins[0] <= cube_mins[0] and cube_maxs[0] <= bowl_maxs[0] and \
           bowl_mins[1] <= cube_mins[1] and cube_maxs[1] <= bowl_maxs[1] and \
           bowl_mins[2] <= cube_mins[2] and \
           cube_maxs[2] <= bowl_maxs[2] + self.cube_half_size.max():
            save_pred_result(
                info["success"] == True, rgb_images,
                text_prompt + f'(env {info["success"]}, pred {True})',
                boxes_filt_batch, pred_phrases_batch, pred_masks_batch
            )
            return 1.0

        save_pred_result(
            info["success"] == False, rgb_images,
            text_prompt + f'(env {info["success"]}, pred {False})',
            boxes_filt_batch, pred_phrases_batch, pred_masks_batch
        )
        return 0.0


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
