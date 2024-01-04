from pathlib import Path

models_dir = Path("models")
models = [x for x in models_dir.iterdir()]

assert len([p for p in models if (p / "acronym_grasps.h5").is_file()]) == 8836

import sapien
import sapien.physx as physx
from sapien.utils import Viewer
print(sapien.__version__)

_engine = sapien.Engine()
_renderer = sapien.SapienRenderer()
_engine.set_renderer(_renderer)

scene_config = sapien.SceneConfig()
physx.set_default_material(static_friction=1.0, dynamic_friction=1.0,
                            restitution=0.0)
scene_config.contact_offset = 0.02
scene_config.solver_iterations = 25
scene_config.solver_velocity_iterations = 1
scene: sapien.Scene = _engine.create_scene(scene_config)
scene.set_timestep(1.0 / 500)
arena = scene.load_widget_from_package("sapien_demo_arena", "DemoArena")

# viewer = Viewer(_renderer)
# viewer.set_scene(scene)
# viewer.paused = True
# viewer.render()

from tqdm import tqdm
for model_dir in tqdm(models):
    obj_category, shapenetid, scale = model_dir.name.split("_")
    scale = float(scale)

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(model_dir / "textured.obj"), scale=[scale] * 3)
    builder.add_multiple_convex_collisions_from_file(str(model_dir / "collision.coacd.ply"), scale=[scale] * 3)
    actor = builder.build()