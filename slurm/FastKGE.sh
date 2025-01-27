#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/FastKGE_WN_CKGE%j.txt
#SBATCH --error=errors/FastKGE_WN_CKGE%j.txt
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=FastKGE_WN_CKGE
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

python main.py -model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 \
-dataset WN_CKGE -learning_rate 1e-1 -using_various_ranks True 