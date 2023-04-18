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
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.geometry import get_axis_aligned_bbox_for_actor, angle_between_vec

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
class PlaceCubeInBowlEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def __init__(self, *args,
                 asset_root: str = None,
                 model_json: str = None,
                 model_ids: List[str] = ('024_bowl'),
                 obj_init_rot_z=True,
                 obj_init_rot=0,
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

        self._check_assets()
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

    def _initialize_actors(self, cube_ori=None, dist_cube_bowl=0.2):
        """cubeA_ori is the angle from bowl to A"""
        self._initialize_bowl_actors()

        if cube_ori is None:
            # cube_ori = self._episode_rng.uniform(-np.pi/4, np.pi*3/4)
            cube_ori = self._episode_rng.uniform(0, 2 * np.pi)
        cube_xy = self.bowl.pose.p[:2] + [np.cos(cube_ori) * dist_cube_bowl,
                                          np.sin(cube_ori) * dist_cube_bowl]

        cube_q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            cube_q = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.cube_half_size[2]
        cube_pose = Pose([cube_xy[0], cube_xy[1], z], cube_q)

        self.cube.set_pose(cube_pose)

    def _initialize_task(self, max_trials=100, verbose=False):
        bowl_pos = self.bowl.pose.p

        self.goal_pos = bowl_pos + [0, 0, 0.05]

    def _get_obs_extra(self) -> OrderedDict:
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
        bowl_up_axis = self.bowl.get_pose().to_transformation_matrix()[:3, :3] \
                     @ z_axis_world
        is_bowl_upwards = abs(angle_between_vec(bowl_up_axis,
                                                z_axis_world)) < 0.1*np.pi
        eval_dict = dict(
            is_cube_grasped=self.agent.check_grasp(self.cube),
            is_cube_inside=is_cube_inside,
            is_robot_static=is_robot_static,
            is_cube_static=is_cube_static,
            is_bowl_static=is_bowl_static,
            is_bowl_upwards=is_bowl_upwards,
            success=(is_cube_inside and is_robot_static and
                     is_cube_static and is_bowl_static and is_bowl_upwards),
        )
        eval_dict.update(self.get_cost())
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

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])
        self._prev_actor_poses = {
            self.cube.name: self.cube.get_pose(),
            self.bowl.name: self.bowl.get_pose(),
        }
