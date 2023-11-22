from copy import deepcopy
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np
import sapien
from sapien import Pose
import sapien.wrapper.urdf_loader
import sapien.physx as physx
from transforms3d.quaternions import mat2quat


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def vectorize_pose(pose: sapien.Pose):
    return np.hstack([pose.p, pose.q])


def set_entity_visibility(entity: sapien.Entity, visibility):
    component = get_render_body_component(entity)
    if component is not None:
        component.visibility = visibility


def hide_entity(entity: sapien.Entity):
    get_render_body_component(entity).disable()


def show_entity(entity: sapien.Entity):
    get_render_body_component(entity).enable()


T = TypeVar('T')
def get_obj_by_name(objs: List[T], name: str, is_unique=True):
    """Get a objects given the name.

    Args:
        objs (List[T]): objs to query. Expect these objects to have a get_name function.
            These may be sapien.Entity, physx.PhysxArticulationLink etc.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        T or List[T]:
            matched T or Ts. None if no matches.
    """
    matched_objects = [x for x in objs if x.get_name() == name]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same name {name}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def get_obj_by_type(objs: List[T], target_type: T, is_unique=True):
    matched_objects = [x for x in objs if type(x) == target_type]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same type {target_type}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def get_render_body_component(entity: sapien.Entity) -> sapien.render.RenderBodyComponent:
    """Get sapien.render.RenderBodyComponent. Assumes entity only ever has one of these 

    Returns: sapien.renderRenderBodyComponent if it exists, None otherwise
    """
    return get_obj_by_type(entity.components, sapien.render.RenderBodyComponent)


def get_rigid_dynamic_component(entity: sapien.Entity) -> physx.PhysxRigidDynamicComponent:
    """Get physx.PhysxRigidDynamicComponent. Assumes entity only ever has one of these 

    Returns: physx.PhysxRigidDynamicComponent if it exists, None otherwise
    """
    return get_obj_by_type(entity.components, physx.PhysxRigidDynamicComponent)


def get_rigid_static_component(entity: sapien.Entity) -> physx.PhysxRigidStaticComponent:
    """Get physx.PhysxRigidStaticComponent. Assumes entity only ever has one of these 

    Returns: physx.PhysxRigidStaticComponent if it exists, None otherwise
    """
    return get_obj_by_type(entity.components, physx.PhysxRigidStaticComponent)


def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for SAPIEN.

    Args:
        urdf_config (dict): dict passed to `sapien.URDFLoader.load`.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
            )

    allowed_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            # In fact, it should support specifying collision-shape-level materials.
            if kk not in allowed_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
                )


def parse_urdf_config(config_dict: dict, scene: sapien.Scene) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = deepcopy(config_dict)

    # Create the global physical material for all links
    mtl_cfg = urdf_config.pop("material", None)
    if mtl_cfg is not None:
        urdf_config["material"] = scene.create_physical_material(**mtl_cfg)

    # Create link-specific physical materials
    materials = {}
    for k, v in urdf_config.pop("_materials", {}).items():
        materials[k] = scene.create_physical_material(**v)

    # Specify properties for links
    for link_config in urdf_config.get("link", {}).values():
        # Substitute with actual material
        link_config["material"] = materials[link_config["material"]]

    return urdf_config


def apply_urdf_config(loader: sapien.wrapper.urdf_loader.URDFLoader, urdf_config: dict):
    # TODO (stao): @fxiang is this complete?
    if "link" in urdf_config:
        for name, link_cfg in urdf_config["link"].items():
            if "material" in link_cfg:
                mat: physx.PhysxMaterial = link_cfg["material"]
                loader.set_link_material(name, mat.static_friction, mat.dynamic_friction, mat.restitution)
            if "patch_radius" in link_cfg:
                loader.set_link_patch_radius(name, link_cfg["patch_radius"])
            if "min_patch_radius" in link_cfg:
                loader.set_link_min_patch_radius(name, link_cfg["min_patch_radius"])
            if "density" in link_cfg:
                loader.set_link_density(name, link_cfg["density"])
            # TODO (stao): throw error if there is a config not used?
    if "material" in urdf_config:
        mat: physx.PhysxMaterial = urdf_config["material"]
        loader.set_material(mat.static_friction, mat.dynamic_friction, mat.restitution)
    if "patch_radius" in urdf_config:
        loader.set_patch_radius(urdf_config["patch_radius"])
    if "min_patch_radius" in urdf_config:
        loader.set_min_patch_radius(urdf_config["min_patch_radius"])
    if "density" in urdf_config:
        loader.set_density(urdf_config["density"])


# -------------------------------------------------------------------------- #
# Entity state
# -------------------------------------------------------------------------- #
def get_actor_state(actor: sapien.Entity):
    pose = actor.get_pose()
    component = get_rigid_dynamic_component(actor)
    if component is None or component.kinematic:
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
    else:
        vel = component.get_linear_velocity()  # [3]
        ang_vel = component.get_angular_velocity()  # [3]
    return np.hstack([pose.p, pose.q, vel, ang_vel])  # [13,]


def set_actor_state(actor: sapien.Entity, state: np.ndarray):
    assert len(state) == 13, len(state)
    actor.set_pose(Pose(state[0:3], state[3:7]))
    component = get_rigid_dynamic_component(actor)
    if component is not None and not component.kinematic:
        component.set_linear_velocity(state[7:10])
        component.set_angular_velocity(state[10:13])


def get_articulation_state(art: physx.PhysxArticulation):
    pose = art.root_pose
    vel = art.root_velocity  # [3]
    ang_vel = art.root_angular_velocity  # [3]
    qpos = art.qpos  # [dof,]
    qvel = art.qvel  # [dof,]
    qacc = art.qacc  # [dof,]
    qf = art.qf  # [dof,]
    drive_target = np.hstack(
        [j.drive_target for j in art.active_joints]
    )  # [dof,]
    drive_vel_target = np.hstack(
        [j.drive_velocity_target for j in art.active_joints]
    )  # [dof,]
    return np.hstack([pose.p, pose.q, vel, ang_vel,
                      qpos, qvel, qacc, qf, drive_target, drive_vel_target])


def set_articulation_state(art: physx.PhysxArticulation, state: np.ndarray):
    art.set_root_pose(Pose(state[0:3], state[3:7]))
    art.set_root_velocity(state[7:10])
    art.set_root_angular_velocity(state[10:13])
    qpos, qvel, qacc, qf, drive_target, drive_vel_target = np.split(state[13:], 6)
    art.set_qpos(qpos)
    art.set_qvel(qvel)
    art.set_qacc(qacc)
    art.set_qf(qf)
    for j, t, v_t in zip(art.active_joints, drive_target, drive_vel_target):
        j.set_drive_target(t)
        j.set_drive_velocity_target(v_t)


def get_articulation_padded_state(articulation: physx.PhysxArticulation, max_dof: int):
    state = get_articulation_state(articulation)
    qpos, qvel = np.split(state[13:], 2)
    nq = len(qpos)
    assert max_dof >= nq, (max_dof, nq)
    padded_state = np.zeros(13 + 2 * max_dof, dtype=np.float32)
    padded_state[:13] = state[:13]
    padded_state[13 : 13 + nq] = qpos
    padded_state[13 + max_dof : 13 + max_dof + nq] = qvel
    return padded_state


# -------------------------------------------------------------------------- #
# Contact
#
# Note that for simplicity, we always compare contact by using entitiy objects
# and check if the entity is the same
# -------------------------------------------------------------------------- #
def get_pairwise_contacts(
    contacts: List[physx.PhysxContact], actor0: sapien.Entity, actor1: sapien.Entity
) -> List[Tuple[physx.PhysxContact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if contact.components[0].entity == actor0 and contact.components[1].entity == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.components[0].entity == actor1 and contact.components[1].entity == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[physx.PhysxContact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first component
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[physx.PhysxContact], actor0: sapien.Entity, actor1: sapien.Entity
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def get_actor_contacts(
    contacts: List[physx.PhysxContact], actor: sapien.Entity
) -> List[Tuple[physx.PhysxContact, bool]]:
    entity_contacts = []
    for contact in contacts:
        if contact.components[0].entity == actor:
            entity_contacts.append((contact, True))
        elif contact.components[1].entity == actor:
            entity_contacts.append((contact, False))
    return entity_contacts


def get_articulation_contacts(
    contacts: List[physx.PhysxContact],
    articulation: physx.PhysxArticulation,
    excluded_entities: Optional[List[sapien.Entity]] = None,
    included_links: Optional[List[physx.PhysxArticulationLinkComponent]] = None,
) -> List[Tuple[physx.PhysxContact, bool]]:
    articulation_contacts = []
    links = articulation.get_links()
    if excluded_entities is None:
        excluded_entities = []
    if included_links is None:
        included_links = links
    for contact in contacts:
        if contact.components[0] in included_links:
            if contact.components[1] in links:
                continue
            if contact.components[1].entity in excluded_entities:
                continue
            articulation_contacts.append((contact, True))
        elif contact.components[1] in included_links:
            if contact.components[0] in links:
                continue
            if contact.components[0].entity in excluded_entities:
                continue
            articulation_contacts.append((contact, False))
    return articulation_contacts


def compute_max_impulse_norm(contact_infos: List[Tuple[physx.PhysxContact, bool]]):
    max_impulse_norms = [0]
    for contact, flag in contact_infos:
        max_impulse_norm = max(
            [np.linalg.norm(point.impulse) for point in contact.points]
        )
        max_impulse_norms.append(max_impulse_norm)
    return max(max_impulse_norms)


def get_articulation_max_impulse_norm(
    contacts: List[physx.PhysxContact],
    articulation: physx.PhysxArticulation,
    excluded_entities: Optional[List[sapien.Entity]] = None,
):
    articulation_contacts = get_articulation_contacts(
        contacts, articulation, excluded_entities
    )
    max_impulse_norm = compute_max_impulse_norm(articulation_contacts)
    return max_impulse_norm


# -------------------------------------------------------------------------- #
# Camera
# -------------------------------------------------------------------------- #
def sapien_pose_to_opencv_extrinsic(sapien_pose_matrix: np.ndarray) -> np.ndarray:
    sapien2opencv = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ex = sapien2opencv @ np.linalg.inv(sapien_pose_matrix)  # world -> camera

    return ex


def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Returns:
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))


def hex2rgba(h, correction=True):
    # https://stackoverflow.com/a/29643643
    h = h.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    rgba = np.array([r, g, b, 1])
    if correction:  # reverse gamma correction in sapien
        rgba = rgba**2.2
    return rgba


def set_render_material(material: sapien.render.RenderMaterial, **kwargs):
    for k, v in kwargs.items():
        if k == "color":
            material.set_base_color(v)
        else:
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: physx.PhysxArticulation, **kwargs):
    for link in articulation.get_links():
        component = get_render_body_component(link.entity)
        if component is None:
            continue
        for s in component.render_shapes:
            if type(s) == sapien.render.RenderShapeTriangleMesh:
                # TODO (stao): do other render shapes permit changing material?
                # TODO (stao): can there be multiple renderbody components, or is it unique?
                for part in s.parts:
                    mat = part.material
                    set_render_material(mat, **kwargs)


# -------------------------------------------------------------------------- #
# Misc
# -------------------------------------------------------------------------- #
def check_joint_stuck(
    articulation: physx.PhysxArticulation,
    active_joint_idx: int,
    pos_diff_threshold: float = 1e-3,
    vel_threshold: float = 1e-4,
):
    actual_pos = articulation.get_qpos()[active_joint_idx]
    target_pos = articulation.get_drive_target()[active_joint_idx]
    actual_vel = articulation.get_qvel()[active_joint_idx]

    return (
        abs(actual_pos - target_pos) > pos_diff_threshold
        and abs(actual_vel) < vel_threshold
    )


def check_actor_static(actor: sapien.Entity, lin_thresh=1e-3, ang_thresh=1e-2):
    component = get_rigid_dynamic_component(actor)
    return (
        np.linalg.norm(component.linear_velocity) <= lin_thresh
        and np.linalg.norm(component.angular_velocity) <= ang_thresh
    )
