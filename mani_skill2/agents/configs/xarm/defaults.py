import numpy as np

from mani_skill2.agents.controllers import (
    deepcopy_dict,
    PDJointPosControllerConfig,
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosMimicControllerConfig,
    PDGripperControllerConfig
)
from mani_skill2.sensors.camera import CameraConfig


class XArmDefaultConfig:
    def __init__(self) -> None:
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                left_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "left_finger_joint",
            "right_finger_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "link_tcp"

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.00053359545815346,  # -10
            0.0453556139430441,   # 850
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[-0.0464982, 0.0200011, 0.0360011],
            q=[-0.70710678, 0, 0.70710678, 0],
            width=128,
            height=128,
            fov=1.57,
            near=0.01,
            far=10,
            entity_uid="xarm_gripper_base_link",
        )


class XArm7DefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_pris_finger.urdf"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]


class XArm7D435DefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_d435.urdf"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e6
        self.arm_damping = 5e4
        self.arm_force_limit = 100
        self.arm_joint_friction = 0.05

        self.gripper_joint_names = ["drive_joint"]
        self.gripper_stiffness = 1e5
        self.gripper_damping = 1e3
        self.gripper_force_limit = 100
        self.gripper_joint_friction = 0.05

        self.ee_link_name = "link_tcp"

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_joint_friction,
            use_delta=True,
            use_target=False,
            interpolate=False,
            normalize_action=True,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_joint_friction,
            ee_link=self.ee_link_name,
            frame="ee",
            use_delta=True,
            use_target=False,
            interpolate=False,
            normalize_action=True,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_bound=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_joint_friction,
            ee_link=self.ee_link_name,
            frame="ee",
            use_delta=True,
            use_target=False,
            interpolate=False,
            normalize_action=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd = PDGripperControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-0.01,  # -10
            upper=0.85,   # 850
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_joint_friction,
            use_delta=False,
            use_target=False,
            interpolate=False,
            normalize_action=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=848,
            height=480,
            fov=np.deg2rad(43.5),
            near=0.01,
            far=10,
            entity_uid="camera_color_frame",
            # entity_uid="camera_depth_frame",
        )


class FloatingXArmDefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm_floating_pris_finger.urdf"
        self.arm_joint_names = [
            "x_axis_joint",
            "y_axis_joint",
            "z_axis_joint",
            "x_rotation_joint",
            "y_rotation_joint",
            "z_rotation_joint",
        ]


class FloatingXArmD435DefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm_floating_pris_finger_d435.urdf"
        self.arm_joint_names = [
            "x_axis_joint",
            "y_axis_joint",
            "z_axis_joint",
            "x_rotation_joint",
            "y_rotation_joint",
            "z_rotation_joint",
        ]

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=848,
            height=480,
            fov=np.deg2rad(43.5),
            near=0.01,
            far=10,
            entity_uid="camera_color_frame",
        )
