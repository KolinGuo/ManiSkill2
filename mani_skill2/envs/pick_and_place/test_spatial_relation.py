from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.visualization.misc import tile_images

from .base_env import StationaryManipulationEnv


@register_env("TestCubeSpatial-v0", max_episode_steps=200)
class TestCubeSpatial(StationaryManipulationEnv):
    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "cameras",
                              "depth_array", "seg_array", "pointcloud")

    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True,
                 #softer_check_grasp=False, static_reward=False,
                 #stage_indicator_obs=False,
                 **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        #self.softer_check_grasp = softer_check_grasp
        #self.static_reward = static_reward
        #self.stage_indicator_obs = stage_indicator_obs
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self._bg_name is None)
        self.cubeA = self._build_cube(self.cube_half_size, color=(1, 0, 0), name="red cube")
        self.cubeB = self._build_cube(self.cube_half_size, color=(0, 1, 0), name="green cube")
        #self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self, cubeB_ori=None, dist_AB=0.1):
        """cubeB_ori is the angle from A to B"""
        xy = np.zeros(2)
        #xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        cubeA_xy = xy
        if cubeB_ori is None:
            cubeB_ori = self._episode_rng.uniform(0, 2 * np.pi)
        cubeB_xy = cubeA_xy + [np.cos(cubeB_ori) * dist_AB, np.sin(cubeB_ori) * dist_AB]

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.cube_half_size[2]
        cubeA_pose = Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi / 2, 0, np.pi * 5 / 8, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.cubeA.pose.p
        #obj_pos = self.obj.pose.p

        ## Sample a goal position far enough from the object
        #for i in range(max_trials):
        #    goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        #    goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
        #    goal_pos = np.hstack([goal_xy, goal_z])
        #    if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
        #        if verbose:
        #            print(f"Found a valid goal at {i}-th trial")
        #        break

        #self.goal_pos = goal_pos
        #self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        return {}
        #obs = OrderedDict(
        #    tcp_pose=vectorize_pose(self.tcp.pose),
        #    goal_pos=self.goal_pos,
        #)
        #if self._obs_mode in ["state", "state_dict"]:
        #    obs.update(
        #        tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
        #        obj_pose=vectorize_pose(self.obj.pose),
        #        tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
        #        obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
        #    )
        #    if self.stage_indicator_obs:
        #        is_obj_placed = self.check_obj_placed()
        #        is_robot_static = self.check_robot_static()
        #        if is_robot_static and is_obj_placed:
        #            stage_indicator = 3
        #        elif is_obj_placed:
        #            stage_indicator = 2
        #        elif self.agent.check_grasp(self.obj):
        #            stage_indicator = 1
        #        else:
        #            stage_indicator = 0
        #        obs.update(stage_indicator=stage_indicator)
        #return obs

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

        #tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        #tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        #cost["tcp_to_obj_dist"] = tcp_to_obj_dist

        #obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        #cost["obj_to_goal_dist"] = obj_to_goal_dist
        return cost

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        #if info["success"]:
        #    reward += 5
        #    return reward

        #tcp_to_obj_dist = info["tcp_to_obj_dist"]
        #reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        #reward += reaching_reward

        #if self.softer_check_grasp:
        #    is_grasped = info["is_obj_grasped"]
        #else:
        #    is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        #reward += 1 if is_grasped else 0.0

        #if is_grasped:
        #    obj_to_goal_dist = info["obj_to_goal_dist"]
        #    place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        #    reward += place_reward

        #    # static reward
        #    if self.static_reward and info["is_obj_placed"]:
        #        qvel = self.agent.robot.get_qvel()[:-2]
        #        static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
        #        reward += static_reward

        return reward

    def _register_render_cameras(self):
        if self.robot_uid == "panda":
            pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10,
                            texture_names=("Color", "Position", "Segmentation"))

    def render_depth_array(self) -> np.ndarray | None:
        """Render function when render_mode='depth_array'"""
        self.update_render()
        images = []
        for camera in self._render_cameras.values():
            pos_img = camera.get_images(take_picture=True)["Position"]
            depth = -pos_img[..., 2]
            depth_image = (depth * 1000.0).astype(np.uint16)
            images.append(depth_image)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        return tile_images(images)

    def render_seg_array(self) -> np.ndarray | None:
        """Render function when render_mode='seg_array'"""
        self.update_render()
        images = []
        for camera in self._render_cameras.values():
            seg_img = camera.get_images(take_picture=True)["Segmentation"]
            seg_img = seg_img[..., 1:2]  # actor_seg
            images.append(seg_img)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        return tile_images(images)

    def render_pointcloud(self) -> dict[str, np.ndarray]:
        """Render function when render_mode='pointcloud'"""
        self.update_render()
        pcds = []
        for camera in self._render_cameras.values():
            pcd = {}
            images = camera.get_images(take_picture=True)

            # construct pcd
            pos_depth = images["Position"]
            mask = pos_depth[..., 3] < 1
            pcd['rgb'] = images["Color"][:, :, :3][mask]
            pcd['xyz'] = pos_depth[:, :, :3][mask]

            # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
            T = camera.camera.get_model_matrix()
            pcd['xyz'] = pcd['xyz'] @ T[:3, :3].T + T[:3, 3]

            pcds.append(pcd)
        fused_pcd = {}
        for key in pcds[0].keys():
            fused_pcd[key] = np.concatenate([pcd[key] for pcd in pcds], axis=0)
        return fused_pcd

    def render(self) -> np.ndarray | dict[str, np.ndarray] | None:
        if self._render_mode == "depth_array":
            return self.render_depth_array()
        elif self._render_mode == "seg_array":
            return self.render_seg_array()
        elif self._render_mode == "pointcloud":
            return self.render_pointcloud()
        else:
            return super().render()

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #
def build_actor_ycb(
    model_id: str, scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: physx.PhysxMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision.coacd.ply")
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
        decomposition="none",
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


@register_env("TestCubeBowlSpatial-v0", max_episode_steps=200)
class TestCubeBowlSpatial(TestCubeSpatial):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def __init__(self, *args,
                 asset_root: str = None,
                 model_json: str = None,
                 model_ids: List[str] = ('024_bowl'),
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

        self.obj_init_rot = obj_init_rot
        self.cube_inside = False

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
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def reset(self, seed=None, reconfigure=False, model_id=None, model_scale=None, cube_inside=False):
        self._set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure

        if cube_inside != self.cube_inside:
            reconfigure = True
        self.cube_inside = cube_inside

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
        self._add_ground(render=self._bg_name is None)
        self._load_model()
        obj_comp = self.obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
        obj_comp.set_linear_damping(0.1)
        obj_comp.set_angular_damping(0.1)
        self.cubeA = self._build_cube(self.cube_half_size, color=(0, 1, 0), name="green cube")
        #self.goal_site = self._build_sphere_site(self.goal_thresh)

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

    def _initialize_actors(self, cubeA_ori=None):
        """cubeA_ori is the angle from bowl to A"""
        self._initialize_bowl_actors()

        if self.cube_inside:
            cubeA_xy = self.obj.pose.p[:2]
        else:
            if cubeA_ori is None:
                cubeA_ori = self._episode_rng.uniform(-np.pi/4, np.pi*3/4)
            cubeA_xy = self.obj.pose.p[:2] + [np.cos(cubeA_ori) * 0.2, np.sin(cubeA_ori) * 0.2]

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.cube_half_size[2] + 0.02
        cubeA_pose = Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)

        self.cubeA.set_pose(cubeA_pose)

try:
    import open3d as o3d
except:
    print("Please install open3d")

def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc

def visualize_point_cloud(points, colors=None, normals=None,
                          show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)
