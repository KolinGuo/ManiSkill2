import numpy as np
import gym
from collections import OrderedDict
from sapien.core import Pose, ActorBase
from sapien.core import renderer as R
from mani_skill2.utils.geometry import rotation_between_vec
from mani_skill2.utils.sapien_utils import get_pairwise_contacts

class SapienViewerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.vectors = OrderedDict()  # {name: R.Node}
        self.contacts = []  # [(actor0, actor1, separate_contact, color)]

        self.models_created = False

    @property
    def viewer(self):
        return self.env.unwrapped._viewer

    def _create_visual_models(self):
        assert self.viewer is not None, "Viewer is not created yet, call render()"
        renderer_context = self.viewer.renderer_context

        self.cone = self.viewer.cone
        self.capsule = self.viewer.capsule

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

        self.models_created = True

    @staticmethod
    def _set_pos_heading_scale(node: R.Node, pos, heading, scale,
                               base_heading=[1.,0.,0.]):
        base_heading, heading = np.array(base_heading), np.array(heading)

        node.set_position(pos)
        node.set_rotation(
            rotation_between_vec(base_heading, heading).as_quat()[[3, 0, 1, 2]]
        )
        node.set_scale(scale)

    def add_vector(self, pos, heading, scale=[0.1,0.1,0.1],
                   color='red', name=None):
        if not self.models_created:
            self._create_visual_models()

        if name is None:
            name = f"vector_{len(self.vectors)}"

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

        self._set_pos_heading_scale(node, pos, heading, scale)
        self.vectors[name] = node

    def toggle_vector(self, name, show: bool):
        for c in self.vectors[name].children:
            c.transparency = 1 - int(show)

    def update_vectors(self, vectors: dict):
        for name, vect in vectors.items():
            if len(vect) == 2:
                pos, heading = vect
                scale = [0.1] * 3
            else:
                pos, heading, scale = vect
            self._set_pos_heading_scale(self.vectors[name], pos, heading, scale)

    def show_contact_visulization(self, actor0: ActorBase, actor1: ActorBase,
                                  separate_contact=True,
                                  scale=[0.1, 0.1, 0.1], color='yellow'):
        """Visualize contact impulse"""
        self.contacts.append((actor0, actor1, separate_contact, scale, color))
        print(f"Adding contact visualization "
              f"between {actor0.name} and {actor1.name} in {color=} {scale=}")

    def update_contact_vectors(self):
        contacts = self.viewer.scene.get_contacts()

        for actor0, actor1, separate_contact, scale, color in self.contacts:
            pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)

            contact_impulses = []
            contact_positions = []

            for contact, flag in pairwise_contacts:
                impulses = np.array(point.impulse for point in contact.points)
                positions = np.array(point.position for point in contact.points)

                if separate_contact:
                    pass
                else:

                    np.linalg.norm(contact_impulses, axis=1, keepdims=True)



                contact_position  # TODO: finish
                pass

            pass
        pass

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self.update_contact_vectors()

        return self.env.render(mode=mode, **kwargs)
