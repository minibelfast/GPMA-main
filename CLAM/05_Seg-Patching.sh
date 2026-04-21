#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at a specified magnification (MAG)
# Typical MAG is 20 (~0.5 MMP); it can also be set to 10 (~1 MMP) or 5 (~2 MMP)
MAG=20
SIZE=256

# Path where CLAM is installed
DIR_REPO=/data2/TCGA-COADREAD/CLAM

# Root path to pathology images 
DIR_READ=/data2/TCGA-COADREAD/slides
DIR_SAVE=/data2/TCGA-COADREAD/patches

cd ${DIR_REPO}

echo "run seg & patching for all slides"
CUDA_VISIBLE_DEVICES=0 python3 /data2/TCGA-COADREAD/CLAM/create_patches_fp.py \
    --source ${DIR_READ} \
    --save_dir ${DIR_SAVE}/tiles-${MAG}x-s${SIZE} \
    --patch_size ${SIZE} \
    --step_size ${SIZE} \
    --preset tcga.csv \
    --patch_magnification ${MAG} \
    --seg --patch --stitch --save_mask \
    --auto_skip --in_child_dir