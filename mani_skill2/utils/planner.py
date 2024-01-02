"""MPlib planner"""
from __future__ import annotations

import mplib
import numpy as np
import sapien.physx as physx
from mplib.pymp.fcl import Box, Capsule, CollisionObject, Convex, Cylinder
from mplib.pymp.planning_world import WorldCollisionResult
from sapien import Entity, Pose
from transforms3d.euler import euler2quat

from mani_skill2.envs.sapien_env import BaseEnv


def get_planner(
    env: BaseEnv,
    move_group: str = "link_tcp",
) -> mplib.Planner:
    """
    Creates an mplib.Planner for the robot in env with all articulations/actors
    as normal_objects. Should be used to create a new planner in env.reconfigure().

    :param move_group: name of robot link to plan. Usually are ["link_eef", "link_tcp"].
    :return planner: created mplib.Planner.
    """
    planner: mplib.Planner = env.agent.robot.get_planner(move_group=move_group)
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
        planning_world.add_normal_object(entity.name, col_obj)
        object_names.append(entity.name)
    planner.object_names = object_names  # save actor names

    return planner


def update_object_pose(planner: mplib.Planner, env: BaseEnv) -> None:
    """Updates planner objects pose (w/o robot) with current environment state"""
    # TODO: handle attached object
    planning_world = planner.planning_world

    for art in env._articulations:
        if art.name == env.agent.robot.name:
            continue
        raise NotImplementedError("Implementation for articulation is not done yet")

    entity_names = [entity.name for entity in env._actors]
    for name in planner.object_names:
        entity: Entity = env._actors[entity_names.index(name)]
        pose: Pose = entity.pose
        col_obj = planning_world.get_normal_object(name)
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
