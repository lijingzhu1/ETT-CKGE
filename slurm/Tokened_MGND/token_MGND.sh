#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/entity%j.txt
#SBATCH --error=errors/entity%j.txt
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=Faster_entity
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

#Relation outperform IncDE 0.199-->0.203
python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 1 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 2 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 3 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 4 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 5 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 6 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 7 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 8 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 9 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -token_num 10 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 5000 10000 10000 10000 



