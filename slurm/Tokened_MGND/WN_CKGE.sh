#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/CKGE%j.txt
#SBATCH --error=errors/CKGE%j.txt
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=faster_WN_CKGE
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE



python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 1


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 2


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 3


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 4


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 5


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 6


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 7


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 8


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 9


python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True \
-use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 \
-learning_rate 0.001 -patience 3 -multi_layer_weight 1 -token_distillation_weight 10000 5000 10000 10000 \
-token_num 10