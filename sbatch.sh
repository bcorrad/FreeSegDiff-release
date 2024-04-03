#!/bin/bash
   
#SBATCH --job-name=openVocab
#SBATCH --nodes=1
#SBATCH --partition=electronic
# #SBATCH --nodelist=kavinsky
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=barbara.corradini@isir.upmc.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# cd /home/corradini/zeroshot_segmentation_coco/script-stable-segmentation
source ~/.bashrc

source activate pro # to activate conda env 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

nohup python -u /home/corradini/FreeSegDiff/script-stable-segmentation/main.py > openVocab.out