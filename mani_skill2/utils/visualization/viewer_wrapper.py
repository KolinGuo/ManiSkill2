import numpy as np
import gym
from collections import OrderedDict
import sapien
from sapien import internal_renderer as R
from mani_skill2.utils.geometry import rotation_between_vec
from mani_skill2.utils.sapien_utils import get_pairwise_contacts


class SapienViewerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.render_nodes = OrderedDict()  # {name: R.Node}
        # [(actor0, actor1, separate_contact, scale, color)]
        self.render_contacts = []

        self.models_created = False

    @property
    def viewer(self):
        return self.env.unwrapped._viewer

    @property
    def dt(self) -> float:
        return self.viewer.scene.timestep

    def _create_visual_models(self):
        assert self.viewer is not None, \
            "Viewer is not created yet, call render()"
        renderer_context = self.viewer.renderer_context

        self.cone = self.viewer.cone
        self.capsule = self.viewer.capsule
        self.sphere = renderer_context.create_capsule_mesh(0.1, 0, 16, 4)

        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

        self.materials = {
            'red': self.viewer.mat_red,
            'green': self.viewer.mat_green,
            'blue': self.viewer.mat_blue,
            'yellow': renderer_context.create_material(
                [0, 0, 0, 1], [1, 1, 0, 1], 0, 0, 0
            ),
            'cyan': self.viewer.mat_cyan,
            'magenta': self.viewer.mat_magenta,
        }

        self.cones = {
            'red': self.viewer.red_cone,
            'green': self.viewer.green_cone,
            'blue': self.viewer.blue_cone,
            'yellow': renderer_context.create_model(
                [self.cone], [self.materials['yellow']]
            ),
            'cyan': renderer_context.create_model(
                [self.cone], [self.materials['cyan']]
            ),
            'magenta': renderer_context.create_model(
                [self.cone], [self.materials['magenta']]
            ),
        }
        self.capsules = {
            'red': self.viewer.red_capsule,
            'green': self.viewer.green_capsule,
            'blue': self.viewer.blue_capsule,
            'yellow': renderer_context.create_model(
                [self.capsule], [self.materials['yellow']]
            ),
            'cyan': self.viewer.cyan_capsule,
            'magenta': self.viewer.magenta_capsule,
        }
        self.spheres = {
            color: renderer_context.create_model(
                [self.sphere], [self.materials[color]]
            ) for color in colors
        }

        self.models_created = True

    @staticmethod
    def _set_pose_scale(node: R.Node, pos=None, heading=None, scale=None,
                        pose=None, base_heading=[1.,0.,0.]):
        if pos is not None:
            assert pose is None, 'pose must be None'
            node.set_position(pos)
        if heading is not None:
            assert pose is None, 'pose must be None'
            base_heading, heading = np.array(base_heading), np.array(heading)
            node.set_rotation(
                rotation_between_vec(base_heading, heading).as_quat()[[3, 0, 1, 2]]
            )
        if pose is not None:
            assert pos is None and heading is None, 'pos and heading must be None'
            node.set_position(pose.p)
            node.set_rotation(pose.q)
        if scale is not None:
            node.set_scale(scale)

    def _add_vector(self, pos, heading, scale=[0.1,0.1,0.1],
                    color='red', name=None):
        if not self.models_created:
            self._create_visual_models()

        if name is None:
            name = f"vector_{len(self.render_nodes)}"

        render_scene: R.Scene = self.viewer.scene.renderer_scene._internal_scene
        node = render_scene.add_node()

        obj = render_scene.add_object(self.cones[color], node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 2
        obj.cast_shadow = False
        obj.transparency = 0

        obj = render_scene.add_object(self.capsules[color], node)
        obj.set_position([0.5, 0, 0])
        obj.set_scale([1.02, 1.02, 1.02])
        obj.shading_mode = 2
        obj.cast_shadow = False
        obj.transparency = 0

        self._set_pose_scale(node, pos, heading, scale)
        self.render_nodes[name] = node

    def toggle_render_node(self, name, show: bool):
        for c in self.render_nodes[name].children:
            c.transparency = 1 - int(show)

    def update_vectors(self, vectors: dict[str, dict]):
        """If not exist, call _add_vector. Else, update vector properties"""
        for name, vector_kwargs in vectors.items():
            if name not in self.render_nodes:
                self._add_vector(name=name, **vector_kwargs)
            else:
                vector_kwargs.pop("color", None)  # cannot update color
                self._set_pose_scale(self.render_nodes[name], **vector_kwargs)
                self.toggle_render_node(name, show=True)

    def _add_point(self, pos, scale=[1,1,1], color='red', name=None):
        if not self.models_created:
            self._create_visual_models()

        if name is None:
            name = f"pt_{len(self.render_nodes)}"

        render_scene: R.Scene = self.viewer.scene.renderer_scene._internal_scene
        node = render_scene.add_node()

        obj = render_scene.add_object(self.spheres[color], node)
        obj.set_scale(scale)
        obj.shading_mode = 2
        obj.cast_shadow = False
        obj.transparency = 0

        self._set_pose_scale(node, pos)
        self.render_nodes[name] = node

    def update_points(self, points: dict[str, dict]={}, scale=np.ones(3)):
        """If not exist, call _add_point. Else, update point properties"""
        scale = self.viewer.axes_scale * scale
        if len(points) == 0:  # update scale based on axes_scale
            for name, node in self.render_nodes.items():
                if name.startswith("pt_"):
                    self._set_pose_scale(node, scale=scale)

        for name, points_kwargs in points.items():
            color = points_kwargs.pop("color", "red")
            points_pos = points_kwargs["pos"].reshape(-1, 3)

            for i, pos in enumerate(points_pos):  # for each point
                pt_name = f"pt_{name}_{i}"

                if pt_name not in self.render_nodes:
                    self._add_point(pos, scale=scale, color=color, name=pt_name)
                else:
                    self._set_pose_scale(self.render_nodes[pt_name],
                                         pos, scale=scale)
                    self.toggle_render_node(pt_name, show=True)

    def show_contact_visualization(self, actor0: sapien.Entity, actor1: sapien.Entity,
                                   draw_separate=True,
                                   scale=[0.05, 0.05, 0.05], color='yellow'):
        """Visualize contact impulse acting on actor0"""
        self.render_contacts.append(
            (actor0, actor1, draw_separate, np.array(scale), color)
        )
        print(f"Adding contact visualization "
              f"between {actor0.name} and {actor1.name} in {color=} {scale=}")

    def update_contact_vectors(self):
        contacts = self.viewer.scene.get_contacts()

        for actor0, actor1, draw_separate, scale, color in self.render_contacts:
            pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)

            contact_positions = []
            contact_impulses = []
            contact_impulse_norms = []

            for contact, flag in pairwise_contacts:
                positions = np.array([point.position for point in contact.points])
                impulses = np.array([point.impulse for point in contact.points])
                # impulse is acting on actor0
                impulses *= 1 if flag else -1
                impulse_norms = np.linalg.norm(impulses, axis=1)

                mask = ~np.isclose(impulse_norms, 0.0)
                if mask.any():
                    positions = positions[mask]
                    impulses = impulses[mask]
                    impulse_norms = impulse_norms[mask]
                else:  # all impulses has zero norm
                    continue

                if draw_separate:
                    contact_positions += [*positions]
                    contact_impulses += [*impulses]
                    contact_impulse_norms += [*impulse_norms]
                else:
                    contact_positions.append(
                        np.sum(positions * impulse_norms[:, None], axis=0) \
                            / np.sum(impulse_norms)
                    )
                    # Add the impulse vectors
                    impulse = np.sum(impulses, axis=0)
                    contact_impulses.append(impulse)
                    contact_impulse_norms.append(np.linalg.norm(impulse))

            # Hide previous contact impulse vectors
            for name in self.render_nodes:
                if name.startswith(f"{actor0.name}_{actor1.name}_"):
                    self.toggle_render_node(name, show=False)

            # Draw impulse vectors
            vectors = {
                f"{actor0.name}_{actor1.name}_{i}": dict(
                    pos=pos, heading=heading, color=color,
                    scale=self.viewer.axes_scale * scale * norm / self.dt
                ) for i, (pos, heading, norm) in enumerate(
                    zip(contact_positions,
                        contact_impulses,
                        contact_impulse_norms)
                )
            }
            self.update_vectors(vectors)

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self.update_contact_vectors()
            self.update_points()

        return self.env.render(mode=mode, **kwargs)
