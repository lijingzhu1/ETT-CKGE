#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/IncDE_%j.txt
#SBATCH --error=errors/IncDE_%j.txt
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=IncDE
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

python main.py -dataset graph_higher -lifelong_name DLKGE_TransE 


python main.py -dataset graph_equal -lifelong_name DLKGE_TransE 

python main.py -dataset graph_lower -lifelong_name DLKGE_TransE 

