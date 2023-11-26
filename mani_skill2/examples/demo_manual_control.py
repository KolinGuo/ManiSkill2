import argparse

import gymnasium as gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.grasping.base_env import GraspingEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode


MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("-s", "--enable-sapien-viewer", action="store_true")
    parser.add_argument("--show-contact", action="store_true")
    parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    if args.reward_mode is not None:
        env: BaseEnv = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            **args.env_kwargs
        )
    else:
        env: BaseEnv = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            **args.env_kwargs
        )
    # Remove TimeLimit wrapper
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    obs, info = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    if args.show_contact:
        from mani_skill2.utils.visualization.viewer_wrapper import SapienViewerWrapper
        env = SapienViewerWrapper(env)

        def update_gripper_direction(env):
            ldirection = env.agent.finger1_link.pose.to_transformation_matrix()[:3, 1]
            rdirection = -env.agent.finger2_link.pose.to_transformation_matrix()[:3, 1]

            vector_kwargs = dict(scale=[0.1] * 3)
            env.update_vectors({
                'ldirection': vector_kwargs | dict(pos=env.agent.finger1_link.pose.p,
                                                   heading=ldirection, color='red'),
                'rdirection': vector_kwargs | dict(pos=env.agent.finger2_link.pose.p,
                                                   heading=rdirection, color='blue'),
            })
        env.show_contact_visualization(env.agent.finger1_link, env.obj, False)
        env.show_contact_visualization(env.agent.finger2_link, env.obj, False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            sapien_viewer = env.render_human()
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1

    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            if args.show_contact:
                update_gripper_direction(env)
            env.render_human()

        render_frame = env.render()

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        key = opencv_viewer.imshow(render_frame)

        if has_base:
            assert args.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
            base_action = np.zeros([4])  # hardcoded
        else:
            base_action = np.zeros([0])

        # Parse end-effector action
        if (
            "pd_ee_delta_pose" in args.control_mode
            or "pd_ee_target_delta_pose" in args.control_mode
        ):
            ee_action = np.zeros([6])
        elif (
            "pd_ee_delta_pos" in args.control_mode
            or "pd_ee_target_delta_pos" in args.control_mode
        ):
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        finish_action = None
        if isinstance(env.unwrapped, GraspingEnv):
            print(f"Grasping Env: {env}", flush=True)
            finish_action = 0

        # Base
        if has_base:
            if key == "w":  # forward
                base_action[0] = 1
            elif key == "s":  # backward
                base_action[0] = -1
            elif key == "a":  # left
                base_action[1] = 1
            elif key == "d":  # right
                base_action[1] = -1
            elif key == "q":  # rotate counter
                base_action[2] = 1
            elif key == "e":  # rotate clockwise
                base_action[2] = -1
            elif key == "z":  # lift
                base_action[3] = 1
            elif key == "x":  # lower
                base_action[3] = -1

        # End-effector
        if num_arms > 0:
            # Position
            if key == "i":  # +x
                ee_action[0] = EE_ACTION
            elif key == "k":  # -x
                ee_action[0] = -EE_ACTION
            elif key == "j":  # +y
                ee_action[1] = EE_ACTION
            elif key == "l":  # -y
                ee_action[1] = -EE_ACTION
            elif key == "u":  # +z
                ee_action[2] = EE_ACTION
            elif key == "o":  # -z
                ee_action[2] = -EE_ACTION

            # Rotation (axis-angle)
            if key == "5":  # +x-axis
                ee_action[3:6] = (1, 0, 0)
            elif key == "2":  # -x-axis
                ee_action[3:6] = (-1, 0, 0)
            elif key == "1":  # +y-axis
                ee_action[3:6] = (0, 1, 0)
            elif key == "3":  # -y-axis
                ee_action[3:6] = (0, -1, 0)
            elif key == "4":  # +z-axis
                ee_action[3:6] = (0, 0, 1)
            elif key == "6":  # -z-axis
                ee_action[3:6] = (0, 0, -1)

        # Gripper
        if has_gripper:
            if key == "f":  # open gripper
                gripper_action = 1
            elif key == "g":  # close gripper
                gripper_action = -1

        # Finish action
        if finish_action is not None:
            if key == "y":
                finish_action = 1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            print("[ INFO ] Switching to SAPIEN viewer")
            render_wait()
            print("[ INFO ] Switching back from SAPIEN viewer")
        elif key == "r":  # reset env
            obs, info = env.reset()
            gripper_action = 1
            after_reset = True
            continue
        elif key == None:  # exit
            break

        # Visualize observation
        if key == "v":
            if "rgbd" in env.obs_mode:
                from itertools import chain

                from mani_skill2.utils.visualization.misc import (
                    observations_to_images,
                    tile_images,
                )

                images = list(
                    chain(*[observations_to_images(x) for x in obs["image"].values()])
                )
                render_frame = tile_images(images)
                opencv_viewer.imshow(render_frame)
            elif "pointcloud" in env.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        if args.env_id in MS1_ENV_IDS:
            action_dict = dict(
                base=base_action,
                right_arm=ee_action,
                right_gripper=gripper_action,
                left_arm=np.zeros_like(ee_action),
                left_gripper=np.zeros_like(gripper_action),
            )
            action = env.agent.controller.from_action_dict(action_dict)
        else:
            action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
            action = env.agent.controller.from_action_dict(action_dict)

        if finish_action is not None:
            print(f"Finish action: {finish_action}", flush=True)
            action = np.hstack([action, finish_action])

        obs, reward, terminated, truncated, info = env.step(action)
        print("reward", reward)
        print("terminated", terminated)
        print("truncated", truncated)
        print("info", info)
        print(env.agent.robot.get_qpos())

    env.close()


if __name__ == "__main__":
    main()
