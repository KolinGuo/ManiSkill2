"""MPlib planner"""
from __future__ import annotations

import mplib
import numpy as np
import sapien.physx as physx
import toppra as ta
from mplib.pymp.articulation import ArticulatedModel
from mplib.pymp.fcl import Box, Capsule, CollisionObject, Convex, Cylinder
from mplib.pymp.planning_world import WorldCollisionResult
from sapien import Entity, Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2.envs.sapien_env import BaseEnv


class Planner(mplib.Planner):
    """
    Wrapper for mplib.Planner to ignore collisions between certain objects
    Written for mplib==0.0.9
    """

    def __init__(self, *args, collision_free_pairs: list[tuple[str]] = [], **kwargs):
        """
        :param collision_free_pairs: always ignore collisions between these link pairs.
                                     list of link name tuples. '*' matches all links.
                                     Only affects collisions of normal_object type.
                                     E.g., [("cube", '*')] ignores all collisions
                                     with cube.
        """
        super().__init__(*args, **kwargs)

        self.collision_free_pairs = collision_free_pairs
        self.collision_ignored_pairs: list[tuple[str]] = []

        # Mask for joints that have equivalent values (revolute joints with range > 2pi)
        self.equiv_joint_mask = [
            t.startswith("JointModelR") for t in self.joint_types
        ] & (self.joint_limits[:, 1] - self.joint_limits[:, 0] > 2 * np.pi)

    @property
    def all_collision_ignored_pairs(self) -> set[tuple[str]]:
        return set(self.collision_free_pairs + self.collision_ignored_pairs)

    def _filter_collisions(
        self, collisions: list[WorldCollisionResult], *, verbose: bool = False
    ) -> list[WorldCollisionResult]:
        """Filter normal_object type collisions"""
        filtered = []
        ignored_pairs = self.all_collision_ignored_pairs
        for collision in collisions:
            if collision.collision_type == "normal_object":
                link_name1 = collision.link_name1
                link_name2 = collision.link_name2
                matched_pairs = [
                    (link_name1, link_name2),
                    (link_name2, link_name1),
                    (link_name1, '*'),
                    ('*', link_name1),
                    (link_name2, '*'),
                    ('*', link_name2),
                ]
                if any(p in ignored_pairs for p in matched_pairs):
                    continue
            filtered.append(collision)
        if verbose and len(filtered) > 0:
            print_collision_info(filtered, with_contact=True)
        return filtered

    # ----- Methods with improvements ----- #
    def check_joint_limit(self, qpos: np.ndarray) -> bool:
        """Check joint limits, always clip q towards lower limits for revolute joints
        i.e., q will be in range [qlimit[0], qlimit[0] + 2*pi)

        :param qpos: joint positions, qpos will be clipped if within_limits.
                     If not within_limits, qpos might not be fully clipped.
        :return within_limits: whether the joint positions are within joint limits.
        """
        for i, (q, joint_type, qlimit) in enumerate(
            zip(qpos, self.joint_types, self.joint_limits)
        ):
            if joint_type.startswith("JointModelR"):  # revolute joint
                if np.abs(q - qlimit[0]) < 1e-3:
                    continue
                q -= 2 * np.pi * np.floor((q - qlimit[0]) / (2 * np.pi))
                qpos[i] = q  # clip qpos
                if q > qlimit[1] + 1e-3:
                    return False
            else:
                if q < qlimit[0] - 1e-3 or q > qlimit[1] + 1e-3:
                    return False
        return True

    # ----- Methods that uses _filter_collisions() ----- #
    def check_for_collision(
        self,
        collision_function,
        articulation: ArticulatedModel = None,
        qpos: np.ndarray = None,
        *,
        verbose: bool = False,
    ) -> list[WorldCollisionResult]:
        """
        :param collision_function: collision function to call.
        :param articulation: robot model. If none will be self.robot.
        :param qpos: robot joint positions. If None will be the current qpos.
        :param verbose: whether to print collision info if any collision exists.
        :return results: list of collisions. Collision exists if the list is not empty.
        """
        return self._filter_collisions(
            super().check_for_collision(collision_function, articulation, qpos),
            verbose=verbose
        )

    def check_for_all_collision(
        self,
        articulation: ArticulatedModel = None,
        qpos: np.ndarray = None,
        *,
        verbose: bool = False,
    ) -> list[WorldCollisionResult]:
        """Check if the robot is in self-collision or collision with the environment.

        :param articulation: robot model. If none will be self.robot.
        :param qpos: robot joint positions. If None will be the current qpos.
        :param verbose: whether to print collision info if any collision exists.
        :return results: list of collisions. Collision exists if the list is not empty.
        """
        return self.check_for_collision(
            self.planning_world.collide_full, articulation, qpos, verbose=verbose
        )

    def check_for_self_collision(
        self,
        articulation: ArticulatedModel = None,
        qpos: np.ndarray = None,
        *,
        verbose: bool = False,
    ) -> list[WorldCollisionResult]:
        """Check if the robot is in self-collision.

        :param articulation: robot model. If none will be self.robot.
        :param qpos: robot joint positions. If None will be the current qpos.
        :param verbose: whether to print collision info if any collision exists.
        :return results: list of collisions. Collision exists if the list is not empty.
        """
        return self.check_for_collision(
            self.planning_world.self_collide, articulation, qpos, verbose=verbose
        )

    def check_for_env_collision(
        self,
        articulation: ArticulatedModel = None,
        qpos: np.ndarray = None,
        *,
        verbose: bool = False,
    ) -> list[WorldCollisionResult]:
        """Check if the robot is in collision with the environment

        :param articulation: robot model. If none will be self.robot.
        :param qpos: robot joint positions. If None will be the current qpos.
        :param verbose: whether to print collision info if any collision exists.
        :return results: list of collisions. Collision exists if the list is not empty.
        """
        return self.check_for_collision(
            self.planning_world.collide_with_others, articulation, qpos, verbose=verbose
        )

    def IK(
        self,
        goal_pose: np.ndarray,
        start_qpos: np.ndarray,
        mask: np.ndarray = [],
        *,
        n_init_qpos: int = 20,
        threshold: float = 0.001,
        return_closest: bool = False,
        verbose: bool = False,
    ) -> tuple[str, list[np.ndarray] | np.ndarray | None]:
        """Compute inverse kinematics

        :param goal_pose: goal pose (xyz, wxyz), (7,) np.floating np.ndarray.
        :param start_qpos: starting qpos, (ndof,) np.floating np.ndarray.
        :param mask: qpos mask to disable planning, (ndof,) bool np.ndarray.
        :param n_init_qpos: number of init qpos to sample.
        :param threshold: distance_6D threshold for marking sampled IK as success.
                          distance_6D is position error norm + quaternion error norm.
        :param return_closest: whether to return the qpos that is closest to start_qpos,
                               considering equivalent joint values.
        :param verbose: whether to print collision info if any collision exists.
        :return status: IK status, "Success" if succeeded.
        :return q_goals: list of sampled IK qpos, (ndof,) np.floating np.ndarray.
                         IK is successful if len(q_goals) > 0.
                         If return_closest, q_goals is np.ndarray if successful
                         and None if not successful.
        """
        move_link_idx = self.link_name_2_idx[self.move_group]
        move_joint_idx = self.move_group_joint_indices
        self.robot.set_qpos(start_qpos, True)

        min_dis = 1e9
        q_goals = []
        qpos = start_qpos
        for _ in range(n_init_qpos):
            ik_qpos, ik_success, ik_error = self.pinocchio_model.compute_IK_CLIK(
                move_link_idx, goal_pose, qpos, mask
            )
            # NOTE: check_joint_limit() will clip qpos towards lower limits
            success = ik_success and self.check_joint_limit(ik_qpos)

            if success:
                # check collision
                self.planning_world.set_qpos_all(ik_qpos[move_joint_idx])
                if (
                    len(self._filter_collisions(
                        self.planning_world.collide_full(), verbose=verbose
                    )) > 0
                ):
                    success = False

            if success:
                self.pinocchio_model.compute_forward_kinematics(ik_qpos)
                new_pose = self.pinocchio_model.get_link_pose(move_link_idx)
                tmp_dis = self.distance_6D(
                    goal_pose[:3], goal_pose[3:], new_pose[:3], new_pose[3:]
                )
                if tmp_dis < min_dis:
                    min_dis = tmp_dis
                if tmp_dis < threshold:
                    for q_goal in q_goals:
                        if (
                            np.linalg.norm(
                                q_goal[move_joint_idx] - ik_qpos[move_joint_idx]
                            ) < 0.1
                        ):
                            break  # not unique ik_qpos
                    else:
                        q_goals.append(ik_qpos)

            qpos = self.pinocchio_model.get_random_configuration()
            qpos[mask] = start_qpos[mask]  # use start_qpos for disabled joints

        if len(q_goals) > 0:
            status = "Success"
        elif min_dis != 1e9:
            status = f"IK Failed! Distance {min_dis} is greater than {threshold=}."
        else:
            status = "IK Failed! Cannot find valid solution."

        if return_closest:
            if len(q_goals) > 0:
                q_goals = np.asarray(q_goals)  # [N, ndof]
                start_qpos = np.asarray(start_qpos)[None]  # [1, ndof]

                # Consider equivalent joint values
                q1 = q_goals[:, self.equiv_joint_mask]  # [N, n_equiv_joint]
                q2 = q1 + 2 * np.pi  # equivalent joints
                start_q = start_qpos[:, self.equiv_joint_mask]  # [1, n_equiv_joint]

                # Mask where q2 is valid and closer to start_q
                q2_closer_mask = (
                    q2 < self.joint_limits[:, 1][None, self.equiv_joint_mask]
                ) & (np.abs(q1 - start_q) > np.abs(q2 - start_q))  # [N, n_equiv_joint]
                q_goals[:, self.equiv_joint_mask] = np.where(q2_closer_mask, q2, q1)

                q_goals = q_goals[np.linalg.norm(q_goals - start_qpos, axis=1).argmin()]
            else:
                q_goals = None
        return status, q_goals

    def plan(
        self,
        goal_pose: np.ndarray,
        current_qpos: np.ndarray,
        mask: np.ndarray = [],
        *,
        time_step: float = 0.1,
        rrt_range: float = 0.1,
        planning_time: float = 1,
        fix_joint_limits: bool = True,
        use_point_cloud: bool = False,
        use_attach: bool = False,
        verbose: bool = False,
    ) -> dict[str, str | np.ndarray | np.float64]:
        """Plan path with RRTConnect

        :param goal_pose: goal pose (xyz, wxyz), (7,) np.floating np.ndarray.
        :param current_qpos: current qpos, (ndof,) np.floating np.ndarray.
        :param mask: qpos mask to disable planning, (ndof,) bool np.ndarray.
        :param time_step: time interval between the generated waypoints.
                          The larger the value, the sparser the output waypoints.
        :param rrt_range: the incremental distance in the RRTConnect algorithm,
                          The larger the value, the sparser the sampled waypoints
                          (before time parameterization).
        :param planning_time: time limit for RRTConnect algorithm, in seconds.
        :param fix_joint_limits: whether to clip the current joint positions
                                 if they are out of the joint limits.
        :param use_point_cloud: whether to avoid collisions
                                between the robot and the point cloud.
        :param use_attach: whether to avoid collisions
                           between the attached tool and the point cloud.
                           Requires use_point_cloud to be True.
        :param verbose: whether to display some internal outputs.
        :return result: A dictionary containing:
                        * status: ik_status if IK failed, "Success" if RRT succeeded.
                        If successful, the following key/value will be included:
                        * time: Time step of each waypoint, (n_step,) np.float64
                        * position: qpos of each waypoint, (n_step, ndof) np.float64
                        * velocity: qvel of each waypoint, (n_step, ndof) np.float64
                        * acceleration: qacc of each waypoint, (n_step, ndof) np.float64
                        * duration: optimal duration of the generated path, np.float64
                        Note that ndof is n_active_dof
        """
        if len(self.all_collision_ignored_pairs) > 0:
            raise NotImplementedError(
                "mplib.pymp.ompl.OMPLPlanner.plan() does not support "
                f"collision_ignored_pairs yet. Got {self.all_collision_ignored_pairs=}"
            )

        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if current_qpos[i] < self.joint_limits[i][0]:
                    current_qpos[i] = self.joint_limits[i][0] + 1e-3
                if current_qpos[i] > self.joint_limits[i][1]:
                    current_qpos[i] = self.joint_limits[i][1] - 1e-3

        self.robot.set_qpos(current_qpos, True)
        collisions = self._filter_collisions(
            self.planning_world.collide_full(), verbose=verbose
        )
        if len(collisions) != 0:
            print("Invalid start state!")
            for collision in collisions:
                print(f"{collision.link_name1} and {collision.link_name2} collide!")

        idx = self.move_group_joint_indices
        ik_status, goal_qpos = self.IK(goal_pose, current_qpos, mask)
        if ik_status != "Success":
            return {"status": ik_status}

        if verbose:
            print("IK results:")
            for i in range(len(goal_qpos)):
                print(goal_qpos[i])

        goal_qpos_ = []
        for i in range(len(goal_qpos)):
            goal_qpos_.append(goal_qpos[i][idx])
        self.robot.set_qpos(current_qpos, True)

        # NOTE: Not working, planner uses planning_world to check collision internally
        status, path = self.planner.plan(
            current_qpos[idx],
            goal_qpos_,
            range=rrt_range,
            verbose=verbose,
            time=planning_time,
        )

        if status == "Exact solution":
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")

            times, pos, vel, acc, duration = self.TOPP(path, time_step)
            return {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
        else:
            return {"status": "RRT Failed. %s" % status}

    def plan_screw(
        self,
        goal_pose: np.ndarray,
        current_qpos: np.ndarray,
        *,
        time_step: float = 0.1,
        qpos_step: float = 0.1,
        use_point_cloud: bool = False,
        use_attach: bool = False,
        verbose: bool = False,
    ) -> dict[str, str | np.ndarray | np.float64]:
        """Plan path with straight-line screw motion

        :param goal_pose: goal pose (xyz, wxyz), (7,) np.floating np.ndarray.
        :param current_qpos: current qpos, (ndof,) np.floating np.ndarray.
        :param time_step: time interval between the generated waypoints.
                          The larger the value, the sparser the output waypoints.
        :param qpos_step: the incremental distance of the joint positions
                          during the path generation (before time paramtertization).
        :param use_point_cloud: whether to avoid collisions
                                between the robot and the point cloud.
        :param use_attach: whether to avoid collisions
                           between the attached tool and the point cloud.
                           Requires use_point_cloud to be True.
        :param verbose: whether to display some internal outputs.
        :return result: A dictionary containing:
                        * status: "Success" if succeeded.
                        If successful, the following key/value will be included:
                        * time: Time step of each waypoint, (n_step,) np.float64
                        * position: qpos of each waypoint, (n_step, ndof) np.float64
                        * velocity: qvel of each waypoint, (n_step, ndof) np.float64
                        * acceleration: qacc of each waypoint, (n_step, ndof) np.float64
                        * duration: optimal duration of the generated path, np.float64
                        Note that ndof is n_active_dof
        """
        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)
        current_qpos = np.copy(current_qpos)
        self.robot.set_qpos(current_qpos, True)

        def pose7D2mat(pose):
            mat = np.eye(4)
            mat[0:3, 3] = pose[:3]
            mat[0:3, 0:3] = quat2mat(pose[3:])
            return mat

        def skew(vec):
            return np.array(
                [
                    [0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0],
                ]
            )

        def pose2exp_coordinate(pose: np.ndarray) -> tuple[np.ndarray, float]:
            def rot2so3(rotation: np.ndarray):
                assert rotation.shape == (3, 3)
                if np.isclose(rotation.trace(), 3):
                    return np.zeros(3), 1
                if np.isclose(rotation.trace(), -1):
                    return np.zeros(3), -1e6
                theta = np.arccos((rotation.trace() - 1) / 2)
                omega = (
                    1
                    / 2
                    / np.sin(theta)
                    * np.array(
                        [
                            rotation[2, 1] - rotation[1, 2],
                            rotation[0, 2] - rotation[2, 0],
                            rotation[1, 0] - rotation[0, 1],
                        ]
                    ).T
                )
                return omega, theta

            omega, theta = rot2so3(pose[:3, :3])
            if theta < -1e5:
                return omega, theta
            ss = skew(omega)
            inv_left_jacobian = (
                np.eye(3) / theta
                - 0.5 * ss
                + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
            )
            v = inv_left_jacobian @ pose[:3, 3]
            return np.concatenate([v, omega]), theta

        self.pinocchio_model.compute_forward_kinematics(current_qpos)
        ee_index = self.link_name_2_idx[self.move_group]
        current_p = pose7D2mat(self.pinocchio_model.get_link_pose(ee_index))
        target_p = pose7D2mat(goal_pose)
        relative_transform = target_p @ np.linalg.inv(current_p)

        omega, theta = pose2exp_coordinate(relative_transform)

        if theta < -1e4:
            return {"status": "screw plan failed."}
        omega = omega.reshape((-1, 1)) * theta

        index = self.move_group_joint_indices
        path = [np.copy(current_qpos[index])]

        while True:
            self.pinocchio_model.compute_full_jacobian(current_qpos)
            J = self.pinocchio_model.get_link_jacobian(ee_index, local=False)
            delta_q = np.linalg.pinv(J) @ omega
            delta_q *= qpos_step / (np.linalg.norm(delta_q))
            delta_twist = J @ delta_q

            flag = False
            if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
                ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
                delta_q = delta_q * ratio
                delta_twist = delta_twist * ratio
                flag = True

            current_qpos += delta_q.reshape(-1)
            omega -= delta_twist

            def check_joint_limit(q):
                n = len(q)
                for i in range(n):
                    if (
                        q[i] < self.joint_limits[i][0] - 1e-3
                        or q[i] > self.joint_limits[i][1] + 1e-3
                    ):
                        return False
                return True

            within_joint_limit = check_joint_limit(current_qpos)
            self.planning_world.set_qpos_all(current_qpos[index])
            collisions = self._filter_collisions(
                self.planning_world.collide_full(), verbose=verbose
            )

            if (
                np.linalg.norm(delta_twist) < 1e-4
                or len(collisions) > 0
                or not within_joint_limit
            ):
                return {"status": "screw plan failed"}

            path.append(np.copy(current_qpos[index]))

            if flag:
                if verbose:
                    ta.setup_logging("INFO")
                else:
                    ta.setup_logging("WARNING")
                times, pos, vel, acc, duration = self.TOPP(np.vstack(path), time_step)
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration,
                }


def get_planner(
    env: BaseEnv,
    move_group: str = "link_tcp",
    collision_free_pairs: list[tuple[str]] = [],
) -> mplib.Planner:
    """
    Creates an mplib.Planner for the robot in env
    with all articulations/actors as normal_objects

    :param move_group: name of robot link to plan. Usually are ["link_eef", "link_tcp"].
    :param collision_free_pairs: always ignore collisions between these link pairs.
                                 list of link name tuples. '*' matches all links.
                                 Only affects collisions of normal_object type.
                                 E.g., [("cube", '*')] ignores all collisions
                                 with cube.
    :return planner: created mplib.Planner.
    """
    planner: Planner = env.agent.robot.get_planner(move_group=move_group)
    planner.collision_free_pairs = collision_free_pairs
    planning_world = planner.planning_world

    for art in env._articulations:
        if art.name == env.agent.robot.name:
            continue
        raise NotImplementedError("Implementation for articulation is not done yet")

    object_names = []
    for entity in env._actors:
        component = entity.find_component_by_type(physx.PhysxRigidBaseComponent)
        assert component is not None, \
            f"No PhysxRigidBaseComponent found in {entity.name}: {entity.components=}"
        pose: Pose = entity.pose

        # Entity should only have 1 collision shape
        assert len(component.collision_shapes) == 1, \
            f"Should only have 1 collision shape, got {component.collision_shapes=}"
        col_shape = component.collision_shapes[0]

        if isinstance(col_shape, physx.PhysxCollisionShapeBox):
            col_geometry = Box(side=col_shape.half_size * 2)
        elif isinstance(col_shape, physx.PhysxCollisionShapeCapsule):
            col_geometry = Capsule(
                radius=col_shape.radius, lz=col_shape.half_length * 2
            )
            # NOTE: physx Capsule has x-axis along capsule height
            # FCL Capsule has z-axis along capsule height
            pose = pose * Pose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(col_shape, physx.PhysxCollisionShapeConvexMesh):
            assert np.allclose(col_shape.scale, 1.0), \
                f"Not unit scale {col_shape.scale}, need to rescale vertices?"
            col_geometry = Convex(
                vertices=col_shape.vertices, faces=col_shape.triangles
            )
        elif isinstance(col_shape, physx.PhysxCollisionShapeCylinder):
            col_geometry = Cylinder(
                radius=col_shape.radius, lz=col_shape.half_length * 2
            )
            # NOTE: physx Cylinder has x-axis along cylinder height
            # FCL Cylinder has z-axis along cylinder height
            pose = pose * Pose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(col_shape, physx.PhysxCollisionShapePlane):
            raise NotImplementedError(
                "Support for Plane collision is not implemented yet in mplib, "
                "need fcl::Plane"
            )
        elif isinstance(col_shape, physx.PhysxCollisionShapeSphere):
            raise NotImplementedError(
                "Support for Sphere collision is not implemented yet in mplib, "
                "need fcl::Sphere"
            )
        elif isinstance(col_shape, physx.PhysxCollisionShapeTriangleMesh):
            # NOTE: see mplib.pymp.fcl.Triangle
            raise NotImplementedError(
                "Support for TriangleMesh collision is not implemented yet."
            )
        else:
            raise TypeError(f"Unknown shape type: {type(col_shape)}")
        col_obj = CollisionObject(col_geometry, pose.p, pose.q)
        planning_world.add_normal_object(col_obj, entity.name)
        object_names.append(entity.name)
    planner.object_names = object_names  # save actor names

    return planner


def update_object_pose(planner: mplib.Planner, env: BaseEnv) -> None:
    """Updates planner objects pose (w/o robot) with current environment state"""
    planning_world = planner.planning_world

    for art in env._articulations:
        if art.name == env.agent.robot.name:
            continue
        raise NotImplementedError("Implementation for articulation is not done yet")

    entity_names = [entity.name for entity in env._actors]
    for name, col_obj in zip(planner.object_names, planning_world.get_normal_objects()):
        entity: Entity = env._actors[entity_names.index(name)]
        pose: Pose = entity.pose
        # NOTE: Convert poses for Capsule/Cylinder
        if isinstance(col_obj.get_collision_geometry(), (Capsule, Cylinder)):
            pose = pose * Pose(q=euler2quat(0, np.pi / 2, 0))
        col_obj.set_transformation(np.hstack((pose.p, pose.q)))


def print_collision_info(
    collisions: list[WorldCollisionResult], with_contact: bool = False
) -> None:
    """Print information about collisions

    :param collisions: list of collisions.
    :param with_contact: also prints contact info.
    """
    print(f"[Planner] Found {len(collisions)} collisions")
    for i, collision in enumerate(collisions):
        print(f"  [{i}]: type={collision.collision_type}, "
              f"objects=({collision.object_name1}, {collision.object_name2}), "
              f"links=({collision.link_name1}, {collision.link_name2})")
        if with_contact:
            with np.printoptions(precision=6, suppress=True):
                for j, contact in enumerate(collision.res.get_contacts()):
                    print(f"     Contact {j}: pos={contact.pos}, "
                          f"normal={contact.normal}, "
                          f"penetration_depth={contact.penetration_depth:.6g}")
