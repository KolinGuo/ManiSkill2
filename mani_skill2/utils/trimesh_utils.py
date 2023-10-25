from typing import List

import numpy as np
import sapien
import sapien.physx as physx
import trimesh
import open3d as o3d

from .sapien_utils import get_render_body_component


def get_component_meshes(component: physx.PhysxRigidBaseComponent):
    """Get component (collision) meshes in the component's frame."""
    meshes = []
    for geom in component.collision_shapes:
        if isinstance(geom, physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * geom.half_size)
        elif isinstance(geom, physx.PhysxCollisionShapeCapsule):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, physx.PhysxCollisionShapeSphere):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, physx.PhysxCollisionShapePlane):
            continue
        elif isinstance(
            geom, (physx.PhysxCollisionShapeConvexMesh)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.get_triangles()
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(geom.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_visual_body_meshes(visual_body: sapien.render.RenderBodyComponent):
    # TODO (stao): update this
    meshes = []
    for render_shape in visual_body.render_shapes:
        # TODO (stao): do other render shapes permit changing material?
        # TODO (stao): can there be multiple renderbody components, or is it unique?
        if type(render_shape) == sapien.render.RenderShapeTriangleMesh:
            for part in render_shape.parts:
                vertices = part.vertices * render_shape.scale  # [n, 3]
                faces = part.triangles
                # faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.apply_transform(render_shape.local_pose.to_transformation_matrix())
                meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: sapien.Entity):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    comp = get_render_body_component(actor)
    if comp is not None:
        meshes.extend(get_visual_body_meshes(comp))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_component_mesh(component: physx.PhysxRigidBaseComponent, to_world_frame=True):
    mesh = merge_meshes(get_component_meshes(component))
    if mesh is None:
        return None
    if to_world_frame:
        T = component.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: sapien.Entity):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation: physx.PhysxArticulation, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_component_mesh(link, True)
        if mesh is None:
            continue
        meshes.append(mesh)
    return meshes


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    assert isinstance(mesh, trimesh.Trimesh), f"Wrong {type(mesh) = }"
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
