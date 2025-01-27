#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/MGND_WO_MLL%j.txt
#SBATCH --error=errors/MGND_WO_MLL%j.txt
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=MGND_WO_MLL
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

#Relation outperform IncDE 0.199-->0.203
python main.py -dataset RELATION -lifelong_name MGND -using_MAE_loss True \
-use_multi_layers False -use_two_stage False -using_mask_weight False -without_multi_layers True \
-MAE_loss_weights 0.0001 0.0001 0.0001 0.0001 

#Entity
python main.py -dataset ENTITY -lifelong_name MGND -using_MAE_loss True -use_multi_layers False -use_two_stage False \
-use_multi_layers False -use_two_stage False -using_mask_weight False -without_multi_layers True \
-MAE_loss_weights 0.0001 0.0001 0.001 0.001

#Hybrid outperform IncDE 0.224-->0.228
python main.py -dataset HYBRID -lifelong_name MGND -using_MAE_loss True -use_multi_layers False -use_two_stage False \
-use_multi_layers False -use_two_stage False -using_mask_weight False -without_multi_layers True \
-MAE_loss_weights 0.0005 0.0005 0.0005 0.0005

#Fact
python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -use_multi_layers False -use_two_stage False \
-use_multi_layers False -use_two_stage False -using_mask_weight False -without_multi_layers True \
-MAE_loss_weights 0.01 0.01 0.01 0.01

# #graph_equal
# python main.py -dataset graph_equal -lifelong_name MGND -using_MAE_loss True \
# -MAE_loss_weights 0.0001 0.0001 0.0001 0.0001

# #graph_higher
# python main.py -dataset graph_higher -lifelong_name MGND -using_MAE_loss True \
# -MAE_loss_weights 0.0001 0.0001 0.00001 0.00001

# #graph_lower
# python main.py -dataset graph_lower -lifelong_name MGND -using_MAE_loss True \
# -MAE_loss_weights 0.0001 0.0001 0.0001 0.0001