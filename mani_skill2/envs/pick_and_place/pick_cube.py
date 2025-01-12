from collections import OrderedDict

import numpy as np
from sapien import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose, hide_entity, show_entity

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=200)
@register_env("PickCube-v1", max_episode_steps=200,
              softer_check_grasp=True, static_reward=True)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True,
                 softer_check_grasp=False, static_reward=False,
                 stage_indicator_obs=False,
                 **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        self.softer_check_grasp = softer_check_grasp
        self.static_reward = static_reward
        self.stage_indicator_obs = stage_indicator_obs
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self._bg_name is None)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
            if self.stage_indicator_obs:
                is_obj_placed = self.check_obj_placed()
                is_robot_static = self.check_robot_static()
                if is_robot_static and is_obj_placed:
                    stage_indicator = 3
                elif is_obj_placed:
                    stage_indicator = 2
                elif self.agent.check_grasp(self.obj):
                    stage_indicator = 1
                else:
                    stage_indicator = 0
                obs.update(stage_indicator=stage_indicator)
        return obs

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        eval_dict = dict(
            is_obj_grasped=self.agent.check_grasp(self.obj),
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )
        eval_dict.update(self.get_cost())
        return eval_dict

    def get_cost(self) -> dict:
        ''' Calculate the current costs and return a dict '''
        cost = {}

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        cost["tcp_to_obj_dist"] = tcp_to_obj_dist

        obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        cost["obj_to_goal_dist"] = obj_to_goal_dist
        return cost

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_obj_dist = info["tcp_to_obj_dist"]
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        if self.softer_check_grasp:
            is_grasped = info["is_obj_grasped"]
        else:
            is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = info["obj_to_goal_dist"]
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.static_reward and info["is_obj_placed"]:
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward

    def render_human(self) -> None:
        """Render function when render_mode='human'"""
        show_entity(self.goal_site)
        ret = super().render_human()
        hide_entity(self.goal_site)
        return ret

    def render_rgb_array(self) -> np.ndarray | None:
        """Render function when render_mode='rgb_array'"""
        show_entity(self.goal_site)
        ret = super().render_rgb_array()
        hide_entity(self.goal_site)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


@register_env("Cost/PickCube-v0", max_episode_steps=200)
@register_env("Cost/PickCube-v1", max_episode_steps=200,
              softer_check_grasp=True, static_reward=True)
class PickCubeCostEnv(PickCubeEnv):
    def __init__(self, *args, tcp_to_obj_dist_thres=0.02,
                 no_check_grasp=False,
                 **kwargs):
        self.tcp_to_obj_dist_thres = tcp_to_obj_dist_thres
        self.no_check_grasp = no_check_grasp
        super().__init__(*args, **kwargs)

    def get_cost(self) -> dict:
        ''' Calculate the current costs (after subtracting threshold) '''
        cost = super().get_cost()

        # tcp_to_obj_dict < 0.02
        cost["cost_tcp_to_obj_dist"] \
            = cost["tcp_to_obj_dist"] - self.tcp_to_obj_dist_thres

        return cost

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        if self.no_check_grasp:
            is_grasped = False
        elif self.softer_check_grasp:
            is_grasped = info["is_obj_grasped"]
        else:
            is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

        reward += 1 if is_grasped else 0.0

        if is_grasped or self.no_check_grasp:
            obj_to_goal_dist = info["obj_to_goal_dist"]
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.static_reward and info["is_obj_placed"]:
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward


@register_env("LiftCube-v0", max_episode_steps=200)
class LiftCubeEnv(PickCubeEnv):
    """Lift the cube to a certain height."""

    goal_height = 0.2

    def _initialize_task(self):
        self.goal_pos = self.obj.pose.p + [0, 0, self.goal_height]
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return self.obj.pose.p[2] >= self.goal_height + self.cube_half_size[2]

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 2.25
            return reward

        # reaching reward
        gripper_pos = self.tcp.pose.p
        obj_pos = self.obj.pose.p
        dist = np.linalg.norm(gripper_pos - obj_pos)
        reaching_reward = 1 - np.tanh(5 * dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

        # grasp reward
        if is_grasped:
            reward += 0.25

        # lifting reward
        if is_grasped:
            lifting_reward = self.obj.pose.p[2] - self.cube_half_size[2]
            lifting_reward = min(lifting_reward / self.goal_height, 1.0)
            reward += lifting_reward

        return reward


@register_env("PickCubeRegion-v0", max_episode_steps=200)
class PickCubeRegionEnv(PickCubeEnv):
    goal_thresh = 0.2
    min_goal_dist = 0.5
    goal_pose = Pose(p=[0, 0.5, 0.2])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self._bg_name is None)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh,
                                                 pose=self.goal_pose)

    def _initialize_task(self, max_trials=100, verbose=False):
        self.goal_pos = self.goal_site.pose.p
        pass
