import time
from multiprocessing import Process, Queue, current_process
from pathlib import Path

import sapien
import sapien.physx as physx
from sapien.utils import Viewer
from tqdm import tqdm

# print(sapien.__version__)


def _load_model(model_dir: Path):
    _engine = sapien.Engine()
    _renderer = sapien.SapienRenderer()
    _engine.set_renderer(_renderer)

    scene_config = sapien.SceneConfig()
    physx.set_default_material(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    scene_config.contact_offset = 0.02
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 1
    scene: sapien.Scene = _engine.create_scene(scene_config)
    scene.set_timestep(1.0 / 500)

    # arena = scene.load_widget_from_package("sapien_demo_arena", "DemoArena")
    # viewer = Viewer(_renderer)
    # viewer.set_scene(scene)
    # viewer.paused = True
    # viewer.render()

    obj_category, shapenetid, scale = model_dir.name.split("_")
    scale = float(scale)

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(model_dir / "textured.obj"), scale=[scale] * 3)
    builder.add_multiple_convex_collisions_from_file(
        str(model_dir / "collision.coacd.ply"), scale=[scale] * 3
    )
    actor = builder.build()
    return actor


def load_model(model_dirs: list, failed_models: Queue):
    for model_dir in tqdm(model_dirs, disable=len(model_dirs) == 1):
        child_proc = Process(target=_load_model, name=f"SubProc {i}", args=(model_dir,))
        child_proc.start()
        child_proc.join()
        if child_proc.exitcode != 0:
            failed_models.put(model_dir)

    print(f"{current_process().name}: Finished loading {len(model_dirs)} meshes")


if __name__ == "__main__":
    import argparse

    default_model_dir = Path("/rl_benchmark/mani_skill2/data/acronym/models")
    default_output_file = (
        default_model_dir.parent / f"failed_models_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    parser = argparse.ArgumentParser(description="Test load model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=default_model_dir,
        help="Model directory.",
    )
    parser.add_argument(
        "--model-dir-from-file",
        type=str,
        default=None,
        help="Read model_dirs from this file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=default_output_file,
        help="Output file path",
    )
    parser.add_argument(
        "--num-proc", type=int, default=32, help="Number of parallel processes to start"
    )
    args = parser.parse_args()

    if args.model_dir_from_file is not None:
        model_dir_from_file = Path(args.model_dir_from_file)
        assert model_dir_from_file.is_file(), f"{model_dir_from_file=}"
        with model_dir_from_file.open("r") as f:
            lines = f.readlines()
        model_dirs = sorted([Path(line.strip()) for line in lines])

        succes_cnt = len([p for p in model_dirs if (p / "acronym_grasps.h5").is_file()])
        assert succes_cnt == len(
            model_dirs
        ), f"Only {succes_cnt} / {len(model_dirs)} are successful"
        print(
            f"Begin to process {len(model_dirs)} models from {args.model_dir_from_file}"
        )
    else:
        assert Path(args.model_dir).is_dir()
        model_dirs = sorted([x for x in Path(args.model_dir).iterdir()])
        assert (
            len([p for p in model_dirs if (p / "acronym_grasps.h5").is_file()]) == 8836
        )

    num_proc = min(len(model_dirs), args.num_proc)
    q, r = divmod(len(model_dirs), num_proc)
    processes, failed_models_queues = [], []
    for i in range(num_proc):
        model_dirs_proc = model_dirs[i * q + min(i, r) : (i + 1) * q + min(i + 1, r)]
        queue = Queue()
        processes.append(
            Process(target=load_model, name=f"Proc {i}", args=(model_dirs_proc, queue))
        )
        failed_models_queues.append(queue)

    print(f"Starting {len(processes)} processes")
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()
    print(f"Finished {len(processes)} processes")

    failed_models = []
    for queue in failed_models_queues:
        while not queue.empty():
            failed_models.append(queue.get())
    print(f"Found {len(failed_models)} failed models:")
    with Path(args.output_file).open("w") as f:
        for model_dir in failed_models:
            print(model_dir)
            print(model_dir, file=f)
