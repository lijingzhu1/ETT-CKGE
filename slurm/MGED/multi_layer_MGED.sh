#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/MGED_with_multi_layer_%j.txt
#SBATCH --error=errors/MGED_with_multi_layer_%j.txt
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=MGED_with_multi_layer
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

python main.py -dataset RELATION -lifelong_name multi_layer_MGED -using_multi_layer_distance_loss True