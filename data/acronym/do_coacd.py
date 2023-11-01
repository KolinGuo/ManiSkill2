#!/usr/bin/env python3
import os
import multiprocessing as mp
import hashlib
import warnings

import trimesh
import coacd

ctx = mp.get_context("spawn")


def get_file_md5(filename):
    with open(filename, "rb") as file_to_check:
        data = file_to_check.read()
        return hashlib.md5(data).hexdigest()


def _run_coacd(
    filename,
    threshold,
    max_convex_hull,
    preprocess_mode,
    preprocess_resolution,
    resolution,
    mcts_nodes,
    mcts_iterations,
    mcts_max_depth,
    pca,
    merge,
    seed,
    verbose,
    paramstr,
    outfile,
):
    if verbose:
        coacd.set_log_level("info")
    else:
        coacd.set_log_level("warn")

    mesh = trimesh.load(filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=merge,
        seed=seed,
    )

    content = sum([trimesh.Trimesh(*m) for m in result]).export(None, "ply")
    content = content.split(b"\n", 2)
    content.insert(2, ("comment " + paramstr).encode("ascii"))
    content = b"\n".join(content)

    with open(outfile, "wb") as f:
        f.write(content)


def do_coacd(
    filename,
    outfile=None,
    threshold=0.05,
    max_convex_hull=-1,
    preprocess_mode="auto",
    preprocess_resolution=30,
    resolution=2000,
    mcts_nodes=20,
    mcts_iterations=150,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    seed=0,
    verbose=False,
):
    md5 = get_file_md5(filename)
    paramstr = (
        f"md5={md5}, "
        f"threshold={threshold:.2f}, "
        f"max_convex_hull={max_convex_hull}, "
        f"preprocess_mode={preprocess_mode}, "
        f"preprocess_resolution={preprocess_resolution}, "
        f"resolution={resolution}, "
        f"mcts_nodes={mcts_nodes}, "
        f"mcts_iterations={mcts_iterations}, "
        f"mcts_max_depth={mcts_max_depth}, "
        f"pca={pca}, "
        f"merge={merge}, "
        f"seed={seed}"
    )

    if outfile is None:
        outfile = filename + ".coacd.ply"
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            success = f.readline() == b"ply\n"
            success = success and f.readline() == b"format binary_little_endian 1.0\n"
            success = (
                success and f.readline().decode("ascii") == f"comment {paramstr}\n"
            )
            if success:
                if verbose:
                    print("using cached decomposition file")
                return outfile

    p = ctx.Process(
        target=_run_coacd,
        args=(
            filename,
            threshold,
            max_convex_hull,
            preprocess_mode,
            preprocess_resolution,
            resolution,
            mcts_nodes,
            mcts_iterations,
            mcts_max_depth,
            pca,
            merge,
            seed,
            verbose,
            paramstr,
            outfile,
        ),
    )
    p.start()
    p.join()

    if p.exitcode == 0 and os.path.exists(outfile):
        return outfile

    # preprocess is already on, fail immediately
    if preprocess_mode == "on":
        warnings.warn(f"coacd failed on {filename}")

    # try again with preprocess on since auto may have issues
    print("coacd failed, trying again with preprocess_mode on")

    p = ctx.Process(
        target=_run_coacd,
        args=(
            filename,
            threshold,
            max_convex_hull,
            "on",
            preprocess_resolution,
            resolution,
            mcts_nodes,
            mcts_iterations,
            mcts_max_depth,
            pca,
            merge,
            seed,
            verbose,
            paramstr,
            outfile,
        ),
    )
    p.start()
    p.join()
    if p.exitcode == 0 and os.path.exists(outfile):
        return outfile

    warnings.warn(f"coacd failed on {filename}")