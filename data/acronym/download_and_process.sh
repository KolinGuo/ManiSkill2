#!/bin/bash

echo -e "Running this script directly is not recommended. Exiting..."
exit 0

# Clone acronym repo
git clone git@github.com:NVlabs/acronym.git

# Download ACRONYM dataset
python3 -m pip install -U gdown
gdown 1zcPARTCQx2oeiKk7a-wdN_CN-RUVX56c
tar -xzvf acronym.tar.gz
mv -v grasps/ acronym_grasps/

# Install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone ShapeNetSem repo from huggingface
git clone git@hf.co:datasets/ShapeNet/ShapeNetSem-archive
cd ShapeNetSem-archive
unzip ShapeNetSem.zip ShapeNetSem-backup/*.txt ShapeNetSem-backup/*.csv ShapeNetSem-backup/models-textures/* ShapeNetSem-backup/models-OBJ/*

# Install Manifold
cd ../
git clone git@github.com:hjwdzh/Manifold.git
cd Manifold
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make
cd ../../

# Process meshes (requires 'trimesh' and 'sapien' packages)
python3 convert_mesh_watertight.py |& tee terminal.log
