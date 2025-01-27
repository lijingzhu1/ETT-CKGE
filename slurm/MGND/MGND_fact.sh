#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/MGND_fact%j.txt
#SBATCH --error=errors/MGND_fact%j.txt
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=MGND_fact
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

#Relation outperform IncDE 0.199-->0.203
python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.005 0.005 0.005 -save_path ./checkpoint/MGND_fact/ 

python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.001 0.005 0.005 -save_path ./checkpoint/MGND_fact/ 

python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.001 0.001 0.005 -save_path ./checkpoint/MGND_fact/ 

python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.01 0.01 0.01 -save_path ./checkpoint/MGND_fact/ 

python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.001 0.01 0.01 -save_path ./checkpoint/MGND_fact/ 

python main.py -dataset FACT -lifelong_name MGND -using_MAE_loss True -mask_ratio 0.6 -multi_layer_weight 0.001 \
-MAE_loss_weights 0.001 0.001 0.001 0.01 -save_path ./checkpoint/MGND_fact/ 