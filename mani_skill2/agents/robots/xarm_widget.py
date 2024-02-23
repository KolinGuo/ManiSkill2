import os
import warnings
from typing import List, Sequence

import mplib
import numpy as np
import sapien
import sapien.physx as physx
from sapien.utils import Viewer

from mani_skill2 import format_path


class XArm7(sapien.Widget):
    def __init__(self, asset_dir=os.path.dirname(__file__)):
        self.robot: physx.PhysxArticulation = None  # type: ignore
        self.init_qpos: np.ndarray = None  # without mimicked joints  # type: ignore
        self.init_all_qpos: np.ndarray = (
            None  # includes mimicked joints  # type: ignore
        )
        self.asset_dir = asset_dir
        self.urdf_path = os.path.join(self.asset_dir, "xarm7_d435.urdf")
        self.srdf_path = os.path.join(self.asset_dir, "xarm7_d435.srdf")

    def load(self, scene: sapien.Scene) -> None:
        if not scene.physx_system.config.enable_tgs:
            warnings.warn(
                "TGS is not enabled in scene. TGS is recommended "
                "for simulating loop joints."
            )
        if (solver_iter := scene.physx_system.config.solver_iterations) < 15:
            warnings.warn(
                f"Solver iteration ({solver_iter}) of this scene "
                "is probably too small for simulating XArm"
            )

        loader = scene.create_urdf_loader()
        loader.set_material(0.3, 0.3, 0.0)
        loader.set_link_material("left_finger", 2.0, 2.0, 0.0)
        loader.set_link_material("right_finger", 2.0, 2.0, 0.0)
        loader.set_link_patch_radius("left_finger", 0.1)
        loader.set_link_patch_radius("right_finger", 0.1)
        loader.set_link_min_patch_radius("left_finger", 0.1)
        loader.set_link_min_patch_radius("right_finger", 0.1)

        self.robot: physx.PhysxArticulation = loader.load(self.urdf_path)
        # NOTE: currently sapien.wrapper.urdf_loader.URDFLoader does not set robot name
        self.robot.name = os.path.basename(self.urdf_path.replace(".urdf", ""))
        for link in self.robot.links:
            link.disable_gravity = True

        self._create_drives(scene)

        self.arm_joints = self.robot.active_joints[:7]
        self.left_gripper_joint = self.robot.find_link_by_name(
            "left_outer_knuckle"
        ).joint
        self.right_gripper_joint = self.robot.find_link_by_name(
            "right_outer_knuckle"
        ).joint

        self.set_arm_pd([1e6] * 7, [5e4] * 7, [100] * 7)
        self.set_gripper_pd(1e5, 1e3, 100)

        self.init_qpos = np.array([0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2, 0.85])
        self.init_all_qpos = np.hstack((self.init_qpos, [self.init_qpos[-1]] * 5))
        self.robot.set_qpos(self.init_all_qpos)  # type: ignore
        self.robot.set_qvel(np.zeros(13))  # type: ignore
        self.set_arm_target(self.init_qpos[:-1])  # type: ignore
        self.set_gripper_target(self.init_qpos[-1])

    def _create_drives(self, scene: sapien.Scene) -> None:
        """Create drive joints and gear joint for gripper"""
        self.robot.set_qpos(np.zeros(self.robot.dof))  # type: ignore

        lik = self.robot.find_link_by_name("left_inner_knuckle")
        lok = self.robot.find_link_by_name("left_outer_knuckle")
        lf = self.robot.find_link_by_name("left_finger")
        T_pw = lf.entity_pose.inv().to_transformation_matrix()
        p_w = lf.entity_pose.p + lik.entity_pose.p - lok.entity_pose.p  # type: ignore
        T_fw = lik.entity_pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(lik, sapien.Pose(p_f), lf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        rik = self.robot.find_link_by_name("right_inner_knuckle")
        rok = self.robot.find_link_by_name("right_outer_knuckle")
        rf = self.robot.find_link_by_name("right_finger")
        T_pw = rf.entity_pose.inv().to_transformation_matrix()
        p_w = rf.entity_pose.p + rik.entity_pose.p - rok.entity_pose.p  # type: ignore
        T_fw = rik.entity_pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(rik, sapien.Pose(p_f), rf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        gear = scene.create_gear(rok, sapien.Pose(), lok, sapien.Pose(q=[0, 0, 0, 1]))  # type: ignore
        gear.gear_ratio = -1
        gear.enable_hinges()

        for l in [lik, lok, lf, rik, rok, rf]:
            for s in l.collision_shapes:
                s.set_collision_groups([1, 1, 2, 0])

    def set_arm_pd(
        self, ps: Sequence[float], ds: Sequence[float], limits: Sequence[float]
    ) -> None:
        for j, p, d, l in zip(self.arm_joints, ps, ds, limits):
            j.set_drive_property(p, d, l, "acceleration")

    def set_arm_target(self, target: Sequence[float]) -> None:
        for t, j in zip(target, self.arm_joints):
            j.set_drive_target(t)

    def set_arm_velocity_target(self, target: Sequence[float]) -> None:
        for t, j in zip(target, self.arm_joints):
            j.set_drive_velocity_target(t)

    def set_gripper_pd(self, p: float, d: float, limit: float) -> None:
        self.left_gripper_joint.set_drive_property(p, d, limit, "acceleration")
        self.right_gripper_joint.set_drive_property(p, d, limit, "acceleration")

    def set_gripper_target(self, target: float) -> None:
        self.left_gripper_joint.set_drive_target(target)
        self.right_gripper_joint.set_drive_target(target)

    def unload(self, scene: sapien.Scene) -> None:
        scene.remove_articulation(self.robot)

    def __getattr__(self, name: str):
        """Used to access useful physx.PhysxArticulation attributes"""
        if self.robot is not None:
            return getattr(self.robot, name)
        else:
            raise RuntimeError(
                f"{self} is not loaded yet. "
                f"Please call scene.load_widget() on it first"
            )

    # ----- mplib.Planner ----- #
    def get_planner(self, move_group: str = "link_tcp") -> mplib.Planner:
        """Creates an mplib.Planner for the robot

        :param move_group: name of robot link to plan.
                           Usually can be ["link_eef", "link_tcp"].
        """
        link_names = [l.name for l in self.links]
        assert move_group in link_names, f'No link named "{move_group}": {link_names=}'

        return mplib.Planner(
            urdf=self.urdf_path,
            srdf=self.srdf_path,
            user_link_names=link_names,
            user_joint_names=[j.name for j in self.robot.active_joints],
            move_group=move_group,
            # vel/acc limits are from Table 1.2 in xArm User Manual v2.0.0
            joint_vel_limits=np.full(7, np.pi),
            joint_acc_limits=np.full(7, np.deg2rad(1145)),
        )

    # ----- Attribute wrappers of physx.PhysxArticulation for mimicked joints ----- #
    def get_dof(self) -> int:
        """Robot degree of freedom"""
        return 8

    dof = property(get_dof)

    def get_active_joints(self) -> List[physx.PhysxArticulationJoint]:
        """Robot active joints"""
        return self.robot.active_joints[:8]

    active_joints = property(get_active_joints)

    def get_qpos(self) -> np.ndarray:
        """Robot joint position"""
        return self.robot.qpos[:8]

    def set_qpos(self, qpos: Sequence[float]) -> None:
        if isinstance(qpos, np.ndarray):
            qpos = qpos.tolist()
        self.robot.qpos = qpos + [qpos[-1]] * 5  # type: ignore

    qpos = property(get_qpos, set_qpos)

    def get_qvel(self) -> np.ndarray:
        """Robot joint velocity"""
        return self.robot.qvel[:8]

    def set_qvel(self, qvel: Sequence[float]) -> None:
        if isinstance(qvel, np.ndarray):
            qvel = qvel.tolist()
        self.robot.qvel = qvel + [qvel[-1]] * 5  # type: ignore

    qvel = property(get_qvel, set_qvel)

    def get_qacc(self) -> np.ndarray:
        """Robot joint acceleration"""
        return self.robot.qacc[:8]

    def set_qacc(self, qacc: Sequence[float]) -> None:
        if isinstance(qacc, np.ndarray):
            qacc = qacc.tolist()
        self.robot.qacc = qacc + [qacc[-1]] * 5  # type: ignore

    qacc = property(get_qacc, set_qacc)

    def get_qf(self) -> np.ndarray:
        """Robot joint force/torque"""
        return self.robot.qf[:8]

    def set_qf(self, qf: Sequence[float]) -> None:
        if len(qf) == 13:  # handle set_qf using results from compute_passive_force
            self.robot.qf = qf  # type: ignore
            return
        if isinstance(qf, np.ndarray):
            qf = qf.tolist()
        self.robot.qf = qf + [qf[-1]] * 5  # type: ignore

    qf = property(get_qf, set_qf)

    def get_qlimit(self) -> np.ndarray:
        """Robot joint limits"""
        return self.robot.qlimit[:8]

    qlimit = property(get_qlimit)

    def get_qlimits(self) -> np.ndarray:
        """Robot joint limits (exactly the same as get_qlimit)"""
        return self.robot.qlimits[:8]

    qlimits = property(get_qlimits)


def main():
    engine = sapien.Engine()
    config = sapien.SceneConfig()
    config.solver_iterations = 20
    scene = engine.create_scene(config)

    scene.load_widget_from_package("sapien_demo_arena", "DemoArena")
    scene.set_timestep(1 / 1200)

    xarm = XArm7(asset_dir=format_path("{PACKAGE_ASSET_DIR}/descriptions/"))
    scene.load_widget(xarm)

    viewer = Viewer()
    viewer.set_scene(scene)

    # ----- XArm no gravity ----- #
    xarm.set_qpos(
        [
            9.216433e-07,
            0.3019557,
            7.410042e-06,
            0.8741246,
            4.4826098e-05,
            0.5723556,
            -1.5709692,
        ]
        + [0.85]
    )
    xarm.set_arm_target([
        9.216433e-07,
        0.3019557,
        7.410042e-06,
        0.8741246,
        4.4826098e-05,
        0.5723556,
        -1.5709692,
    ])
    xarm.set_gripper_target(0.85)

    b = scene.create_actor_builder()
    size = [0.015, 0.015, 0.015]
    b.add_box_collision(half_size=size, density=1000)  # type: ignore
    b.add_box_visual(half_size=size)  # type: ignore
    box = b.build()
    box.set_pose(sapien.Pose([0.420, -0.2, 0.015]))  # type: ignore
    drive = scene.create_drive(None, sapien.Pose(), box, sapien.Pose())
    drive.set_drive_property_y(1e2, 1e2)
    # drive.drive_target = sapien.Pose([0.425, 0.07, 0.015])
    drive.set_drive_velocity_target([0, 1, 0], [0, 0, 0])

    camera = scene.add_camera(
        name="test", width=3693, height=2806, fovy=1.57, near=0.1, far=1e03
    )
    camera.set_local_pose(
        sapien.Pose(
            [-0.345796, -0.305399, 0.848966],  # type: ignore
            [0.926578, 0.00304896, 0.37601, -0.00779426],  # type: ignore
        )
    )

    viewer.paused = True
    count = 0
    while not viewer.closed:
        viewer.render()
        scene.step()
        # xarm.set_gripper_target((np.sin(count / 100) + 1) * 0.4)
        # xarm.set_gripper_target(0.85)
        count += 1


if __name__ == "__main__":
    main()
