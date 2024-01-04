#!/bin/bash

for x in {1..5}; do
  logfile_name="terminal_coacd_$(date "+%Y%m%d_%H%M%S").log"
  # Process meshes (requires 'trimesh' and 'sapien' packages)
  python3 convert_mesh_watertight.py |& tee "$logfile_name"

  echo "Succeed cnt: $(find ./models -name "acronym_grasps*" | wc -l)" >> "$logfile_name"
done
