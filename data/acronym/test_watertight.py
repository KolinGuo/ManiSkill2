from pathlib import Path

import trimesh
from tqdm import tqdm


def is_mesh_watertight(file_path):
    # Load the mesh from the given file path
    # mesh = trimesh.load(file_path, force="mesh")
    mesh: trimesh.base.Trimesh = trimesh.load(file_path, process=False, force="mesh")  # type: ignore
    # Check if the mesh is watertight
    return mesh.is_watertight


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test if meshes are watertight",
    )

    root_dir = Path(__file__).resolve().parent
    default_data_dir = root_dir / "models"

    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Directory of processed object models",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"{data_dir=}"
    mesh_filename = "collision.coacd.ply"

    grasp_obj_paths = list(data_dir.glob("**/acronym_grasps.h5"))
    num_grasp_objs = len(grasp_obj_paths)
    print(f"Found {num_grasp_objs} grasping object files")

    for grasp_obj_path in tqdm(grasp_obj_paths):
        mesh_path = grasp_obj_path.parent / mesh_filename
        assert mesh_path.is_file(), f"{mesh_path=}"

        if not is_mesh_watertight(mesh_path):
            print(f"ERROR: Output mesh is not watertight '{mesh_path}'")
            continue
