#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/MGED_%j.txt
#SBATCH --error=errors/MGED_%j.txt
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=Relation_MGED
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

python main.py -dataset RELATION -lifelong_name MGED -using_MAE_loss True -two_stage_epoch_num 5 \
-MAE_loss_weights 0.0001 0.0001 0.0001 0.0001


python main.py -dataset ENTITY -lifelong_name MGED -using_MAE_loss True -two_stage_epoch_num 5 \
-MAE_loss_weights 0.00001 0.00001 0.00001 0.00001

