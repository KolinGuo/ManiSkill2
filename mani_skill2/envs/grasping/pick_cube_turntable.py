from collections import OrderedDict

import numpy as np
import sapien.physx as physx
from sapien import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.camera import resize_obs_images
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at

from .base_env import GraspingEnv


@register_env("PickCubeTurntable-v0", max_episode_steps=50,
              reward_mode="normalized_dense", control_mode="pd_ee_delta_pose",
              use_random_camera_pose=True)
class PickCubeTurntableEnv(GraspingEnv):
    def __init__(self, *args,
                 cube_init_rot_z=True,
                 turntable_radius: float = 0.144,  # 28.8 cm diameter
                 turntable_half_length: float = 0.026,  # 5.2 cm height
                 turntable_rot_speed=None,  # rad/s
                 **kwargs):
        self.cube_init_rot_z = cube_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        self.turntable_radius = turntable_radius
        self.turntable_half_length = turntable_half_length
        self.turntable_rot_speed = turntable_rot_speed

        self.goal_thresh = self.cube_half_size.max() * 2

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------- #
    # Load model
    # ---------------------------------------------------------------------- #
    def _load_actors(self):
        self.ground = self._add_table_as_ground()

        self.turntable = self._build_turntable(self.turntable_radius,
                                               self.turntable_half_length)
        self.turntable_comp = self.turntable.find_component_by_type(
            physx.PhysxRigidDynamicComponent
        )
        self.cube = self._build_cube(self.cube_half_size, color=(1, 0, 0), name="cube")

    # ---------------------------------------------------------------------- #
    # Reset (Env Initialization)
    # ---------------------------------------------------------------------- #
    def _initialize_cube_actor(self):
        """Returns if initialization succeeds or times out"""
        r = self._episode_rng.uniform(
            0.0, self.turntable_radius - np.linalg.norm(self.cube_half_size[:2])
        )
        cube_ori = self._episode_rng.uniform(0, 2 * np.pi)
        cube_x = self.turntable.pose.p[0] + np.cos(cube_ori) * r
        cube_y = self.turntable.pose.p[1] + np.sin(cube_ori) * r
        cube_z = self.cube_half_size[2] + self.turntable_half_length * 2

        cube_q = [1, 0, 0, 0]
        if self.cube_init_rot_z:
            cube_q = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cube_pose = Pose([cube_x, cube_y, cube_z], cube_q)

        self.cube.set_pose(cube_pose)

    def _initialize_actors(self):
        self._initialize_turntable_actor(self.turntable_rot_speed)
        self._initialize_cube_actor()

    def _check_object_visible(self) -> bool:
        """Check if all objects are visible in at least one camera view"""
        image_obs = self._get_obs_images()["image"]
        for cam_name, cam_capture in image_obs.items():
            seg_mask = cam_capture["Segmentation"][..., 1]

            cube_px_cnt = (seg_mask == self.cube.per_scene_id).sum()
            turntable_px_cnt = (seg_mask == self.turntable.per_scene_id).sum()
            if cube_px_cnt >= 100 and turntable_px_cnt >= 2500:
                return True
        return False

    def _sample_tcp_pose(self, max_trials=100) -> bool:
        """Sample a valid tcp pose with objects visible"""
        pose_cam_ee = Pose(np.array(
            [[0, 0, 1, 0],
             [0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 0, 1]]
        ))
        for _ in range(max_trials):
            delta_tcp_pose = Pose(
                q=euler2quat(*np.deg2rad(self._episode_rng.uniform([-10, -10, -180],
                                                                   [10, 10, 180])))
            )
            pose_world_ee = look_at(
                eye=self._episode_rng.uniform([0, -0.4, 0.2], [0.6, 0, 0.8]),
                target=self.cube.pose.p
            ) * pose_cam_ee * delta_tcp_pose

            qpos, success, error = self.pmodel.compute_inverse_kinematics(
                self.ee_link_idx,
                self.agent.robot.pose.inv() * pose_world_ee,
                initial_qpos=np.zeros_like(self.qmask, dtype=np.float32),
                active_qmask=self.qmask,
                max_iterations=100
            )
            if success:
                qpos = qpos[:self.agent.robot.dof]
                qpos[-1] = 0.85  # open gripper

            if (
                success  # feasible IK
                and not (self.check_collision_during_IK  # check collision
                         and self._check_collision(qpos))  # has collision
                and self._check_object_visible()  # objects are visible
            ):
                self.agent.reset(qpos)
                return True
        return False

    def _initialize_agent(self, max_trials=20):
        super()._initialize_agent()

        # Reset camera pose to original
        if self.use_random_camera_pose:
            for cam_name, camera in self._cameras.items():
                camera.camera.local_pose = camera.camera_cfg.pose

        # ----- Check IK feasible and object visible -----
        for _ in range(max_trials):
            # Ensure goal_ee_pose is feasible
            for _ in range(max_trials):
                if self._check_feasible_grasp_pose(self.tcp_goal_pos):
                    break
                print("[ENV] No successful goal grasp pose found!")
                self._initialize_actors()  # reinitialize actors pose
                super()._initialize_agent()  # reset robot qpos and base_pose
            else:
                raise RuntimeError(
                    "Cannot sample layout with valid goal grasp pose.\n"
                    f"Env state: {self.get_state().tolist()}"
                )

            # Ensure init_ee_pose is feasible
            #   and all objects are visible in at least one orig camera view
            for _ in range(max_trials):
                if (self._check_feasible_grasp_pose(self.cube.pose.p)
                        and self._sample_tcp_pose()):
                    break
                print("[ENV] No successful init grasp pose found!")
                self._initialize_cube_actor()  # reinitialize cube pose
            else:
                print("[ENV] Timeout sampling cube pose, resample layout!")
                self._initialize_actors()  # reinitialize bowl & cube pose
                super()._initialize_agent()  # reset robot qpos and base_pose
                continue  # go back to ensure goal_ee_pose
            break
        else:
            raise RuntimeError("Cannot sample valid layout\n"
                               f"Env state: {self.get_state().tolist()}")

        # Randomize camera pose while ensuring
        #   all objects are visible in at least one camera view
        if (self.use_random_camera_pose and not self.random_camera_pose_per_step):
            for _ in range(max_trials):
                self._randomize_camera_pose()
                if self._check_object_visible():
                    break
                print("[ENV] not all objects are visible!")
            else:
                raise RuntimeError(
                    "Cannot sample random camera pose with all objects "
                    f"visible.\nEnv state: {self.get_state().tolist()}"
                )

    # ---------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------- #
    def get_obs(self) -> OrderedDict:
        """Wrapper for get_obs()"""
        obs = super().get_obs()

        if self._obs_mode == "image":
            # Keep only target object segmentation: (H, W, 1) bool
            for cam_name, cam_obs in obs["image"].items():
                cam_obs["Segmentation"] = (
                    cam_obs["Segmentation"][..., [1]] == self.cube.per_scene_id
                )
            obs = resize_obs_images(obs, self.image_obs_shape)
        return obs

    # ---------------------------------------------------------------------- #
    # Reward mode
    # ---------------------------------------------------------------------- #
    def compute_dense_reward(self, info, **kwargs) -> float:
        reward = 0.0

        tcp_to_cube_dist = info["tcp_to_cube_dist"]
        reaching_reward = 1 - np.tanh(5 * tcp_to_cube_dist)
        reward += reaching_reward

        if info["is_cube_grasped"]:
            reward += 1

        return reward

    def compute_final_reward(self, obs, info) -> float:
        """Computes the final reward after performing ending manipulation trajectory"""
        if info["success"]:
            return 5.0
        else:
            return 3.0

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 5.0

    def compute_normalized_final_reward(self, **kwargs):
        return self.compute_final_reward(**kwargs) / 5.0

    # ---------------------------------------------------------------------- #
    # Step
    # ---------------------------------------------------------------------- #
    def _before_simulation_step(self):
        """For simulating dynamic scene, called inside `step_action()`"""
        self._update_turntable_actor()

    def execute_ending_action(self) -> tuple:
        """Performs ending manipulation trajectory after policy outputs finish signal
        If object is not grasped at the start of this function, no ending action
            will be executed. Reward is from compute_normalized_dense_reward()
        Otherwise, reward is from compute_normalized_final_reward()
        Outputs obs, reward, True, info
        """
        if not self.agent.check_grasp(self.cube):
            obs = self.get_obs()
            info = self.get_info(obs=obs)
            info["total_ending_steps"] = 0
            info["ending_static_steps"] = 0
            reward = self.compute_normalized_dense_reward(obs=obs, info=info)
            return obs, reward, True, info

        # ----- Sample valid goal_pos ----- #
        # target_tcp_pose = Pose(p=self.tcp_goal_pos, q=self.tcp.pose.q)
        # goal_pos = target_tcp_pose.p
        # for _ in range(max_trials):
        #     if self._check_feasible_grasp_pose(goal_pos):
        #         break
        #     print("[ENV] No successful goal grasp pose found during ending action!")
        #     goal_pos = (
        #         Pose(self._episode_rng.uniform([-0.1, -0.1, -0.05], [0.1, 0.1, 0.05]))
        #         * target_tcp_pose
        #     ).p
        # else:
        #     raise RuntimeError("Cannot sample valid goal grasp pose.\n"
        #                        f"Env state: {self.get_state().tolist()}")

        # ----- Move TCP to goal_pos via set_action ----- #
        # total_ending_steps = 0
        # for _ in range(100):
        #     delta_tcp_pose = self.tcp.pose.inv() * Pose(p=self.tcp_goal_pos,
        #                                                 q=self.tcp.pose.q)
        #     action = np.asarray(delta_tcp_pose.p.tolist() + [0, 0, 0, -1],
        #                         dtype=np.float32)
        #     action[:3] *= 10.0  # -0.1 => -1
        #     self.step_action(action)
        #     print(f"Ending action: {action=}")
        #     total_ending_steps += 1

        #     print(f"Ending TCP: {self.tcp.pose=}", flush=True)
        #     print("Joint Target:", [(j.name, j.drive_target[0])
        #                             for j in self.agent.robot.active_joints])
        #     if np.linalg.norm(self.tcp_goal_pos - self.tcp.pose.p) <= 1e-3:  # 1mm
        #         break
        # else:
        #     raise RuntimeError(f"Took too long to reach {self.tcp_goal_pos=}")

        # ----- Move TCP to goal_pos (fastest) ----- #
        target_tcp_pose = self.agent.robot.pose.inv() * Pose(p=self.tcp_goal_pos,
                                                             q=self.tcp.pose.q)
        cur_robot_qpos = (self.agent.robot.robot.qpos if self._wrapped_robot
                          else self.agent.robot.qpos)
        qpos, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx, target_tcp_pose,
            initial_qpos=cur_robot_qpos,
            active_qmask=self.qmask,
            max_iterations=100
        )
        assert success, f"Failed to sample {target_tcp_pose=}"
        self.agent.robot.set_arm_target(qpos[:7])
        self.agent.robot.set_gripper_target(-0.01)

        total_ending_steps = 0
        for _ in range(100):
            self.step_action(None)
            total_ending_steps += 1

            if np.linalg.norm(self.tcp_goal_pos - self.tcp.pose.p) <= 1e-3:  # 1mm
                break
        else:
            raise RuntimeError(f"Took too long to reach {self.tcp_goal_pos=}")

        # ----- Hold TCP static at goal_pos ----- #
        assert self.control_mode == "pd_ee_delta_pose", f"Wrong {self.control_mode=}"

        ending_static_steps = 0
        for _ in range(100):
            action = np.asarray([0] * 6 + [-1], dtype=np.float32)
            self.step_action(action)
            total_ending_steps += 1
            ending_static_steps += 1

            if self.check_robot_static():
                break
        else:
            raise RuntimeError("Took too long to stop robot")

        obs = self.get_obs()
        info = self.get_info(obs=obs)
        info["total_ending_steps"] = total_ending_steps
        info["ending_static_steps"] = ending_static_steps
        reward = self.compute_normalized_final_reward(obs=obs, info=info)
        return obs, reward, True, info

    def check_cube_placed(self):
        return np.linalg.norm(self.tcp_goal_pos - self.cube.pose.p) <= self.goal_thresh

    def evaluate(self, **kwargs):
        is_cube_grasped = self.agent.check_grasp(self.cube)
        is_cube_placed = self.check_cube_placed()
        is_robot_static = self.check_robot_static()

        tcp_to_cube_dist = np.linalg.norm(self.cube.pose.p - self.tcp.pose.p)
        cube_to_goal_dist = np.linalg.norm(self.tcp_goal_pos - self.cube.pose.p)
        tcp_to_goal_dist = np.linalg.norm(self.tcp_goal_pos - self.tcp.pose.p)

        eval_dict = dict(
            is_cube_grasped=is_cube_grasped,
            is_cube_placed=is_cube_placed,
            is_robot_static=is_robot_static,
            success=is_cube_grasped and is_cube_placed and is_robot_static,
            tcp_to_cube_dist=tcp_to_cube_dist,
            cube_to_goal_dist=cube_to_goal_dist,
            tcp_to_goal_dist=tcp_to_goal_dist,
        )
        return eval_dict