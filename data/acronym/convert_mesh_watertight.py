import os
import re
import shutil
from pathlib import Path
from multiprocessing import Process, current_process

import trimesh
from do_coacd import do_coacd


def process_meshes(grasp_h5_paths: list[Path], mesh_dir: Path,
                   output_dir: Path, script_dir: Path = None, coacd_params={}):
    """Run manifold & simplify on meshes and copy corresponding *.mtl files"""
    texture_dir = mesh_dir.parents[1] / "models-textures/textures"
    assert texture_dir.is_dir(), f"{texture_dir=}"

    use_coacd = True
    if script_dir is not None:
        manifold_path = script_dir / "manifold"
        simplify_path = script_dir / "simplify"
        use_coacd = False

    for grasp_h5_path in grasp_h5_paths:
        obj_category, shapenetid, scale = grasp_h5_path.name.split("_")

        mesh_path = mesh_dir / f"{shapenetid}.obj"
        mesh_mtl_path = mesh_dir / f"{shapenetid}.mtl"
        assert mesh_path.is_file(), f"{mesh_path=} does not exist"
        assert mesh_mtl_path.is_file(), f"{mesh_mtl_path=} does not exist"

        save_path = output_dir / grasp_h5_path.stem
        out_mesh_path = save_path / (
            "collision.coacd.ply" if use_coacd else "collision.obj"
        )
        assert not (save_path / "acronym_grasps.h5").is_file(), \
            f"Already processed mesh for {shapenetid=}"

        # Clear directory content
        shutil.rmtree(save_path)
        save_path.mkdir()

        # Copy mesh and meterial file
        shutil.copy(mesh_path, save_path / "textured.obj")
        shutil.copy(mesh_mtl_path, save_path)

        # Copy texture files
        mtl_lines = mesh_mtl_path.open("r").readlines() + mesh_path.open("r").readlines()
        texture_file_names = set([res[0] for line in mtl_lines
                                  if (res := re.findall("^.* (.*.jpg)", line))])
        for texture_name in texture_file_names:
            shutil.copy(texture_dir / texture_name, save_path)

        # Process mesh (with Manifold)
        # if script_dir is not None:
        #     tmp_obj_path = save_path / "temp.watertight.obj"
        #     os.system(f'{manifold_path} {mesh_path} {tmp_obj_path} -s')
        #     os.system(f'{simplify_path} -i {tmp_obj_path} -o {out_mesh_path} -m -r 0.02')
        #     tmp_obj_path.unlink()

        # When failed to process using Manifold, switch to use CoACD
        if not out_mesh_path.is_file():
            _ = do_coacd(str(mesh_path), outfile=str(out_mesh_path), **coacd_params)
        if not out_mesh_path.is_file():
            print(f"ERROR: Failed for {shapenetid=}")
            continue

        # Check if watertight
        mesh = trimesh.load(out_mesh_path, process=False)
        if not mesh.is_watertight:
            print(f"ERROR: Output mesh is not watertight {shapenetid=}")
            continue

        # Copy grasp data
        shutil.copy(grasp_h5_path, save_path / "acronym_grasps.h5")

    print(f"{current_process().name}: Finished processing {len(grasp_h5_paths)} meshes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Conver ShapeNetSem dataset to watertight"
    )

    root_dir = Path(__file__).resolve().parent
    default_grasp_dir = root_dir / "acronym_grasps"
    default_model_obj_dir = root_dir / "ShapeNetSem-archive/ShapeNetSem-backup/models-OBJ/models"
    default_output_dir = root_dir / "models"
    # default_script_dir = root_dir / "Manifold/build"

    parser.add_argument("--grasp-dir", type=str, default=default_grasp_dir,
                        help="Directory of ACRONYM grasps")
    parser.add_argument("--mesh-dir", type=str, default=default_model_obj_dir,
                        help="Directory models-OBJ path")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory path")
    # parser.add_argument("--script-dir", type=str, default=default_script_dir,
    #                     help="Directory of scripts to create watertight meshes")
    parser.add_argument("--num-proc", type=int, default=10,
                        help="Number of parallel processes to start")

    args = parser.parse_args()

    grasp_dir = Path(args.grasp_dir)
    assert grasp_dir.is_dir(), f"{grasp_dir=}"
    mesh_dir = Path(args.mesh_dir)
    assert mesh_dir.is_dir(), f"{mesh_dir=}"
    # script_dir = Path(args.script_dir)
    # assert (script_dir / "manifold").is_file(), f"No manifold in {script_dir=}"
    # assert (script_dir / "simplify").is_file(), f"No simplify in {script_dir=}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grasp_obj_paths = list(grasp_dir.glob("*.h5"))
    num_grasp_objs = len(grasp_obj_paths)
    print(f"Found {num_grasp_objs} grasping object files")

    if output_dir.is_dir():
        processed_grasp_obj_names = [
            p.parent.name for p in output_dir.glob("**/acronym_grasps.h5")
        ]
        grasp_obj_paths = [
            p for p in grasp_obj_paths if p.stem not in processed_grasp_obj_names
        ]
        num_grasp_objs = len(grasp_obj_paths)
        print(f"Found {num_grasp_objs} unprocessed grasping object files")

    num_objs_each = (num_grasp_objs // args.num_proc)
    processes = []
    for i in range(args.num_proc):
        if i == args.num_proc-1:
            obj_paths_proc = grasp_obj_paths[num_objs_each*i:]
        else:
            obj_paths_proc = grasp_obj_paths[num_objs_each*i:num_objs_each*(i+1)]
        processes.append(
            Process(target=process_meshes, name=f"Proc {i}",
                    args=(
                        obj_paths_proc,
                        mesh_dir,
                        output_dir,
                        # script_dir
                    ),
                    kwargs=dict(coacd_params={
                        "threshold": 0.03,
                        "max_convex_hull": 0,
                        "preprocess_resolution": 100,
                        "verbose": True,
                    }))
        )

    print(f"Starting {len(processes)} processes")
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()
    print(f"Finished {len(processes)} processes")
