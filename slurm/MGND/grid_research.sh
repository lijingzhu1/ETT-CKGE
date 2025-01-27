#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/MGND_grid_search_hybrid%j.txt
#SBATCH --error=errors/MGND_grid_search_hybrid%j.txt
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --job-name=MGND_grid_search_hybrid
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.6.1
source activate IncDE
cd /users/PCS0256/lijing/IncDE

# Define the parameter values for the grid search
mask_ratios=(0.3 0.4)
multi_layer_weights=(1 0.1 0.001 0.0001)
without_two_stage=(True)

# Create a directory to save results
save_dir="./checkpoint/MGND/"
mkdir -p $save_dir

# Loop through all combinations of parameters
for mask_ratio in "${mask_ratios[@]}"; do
  for weight in "${multi_layer_weights[@]}"; do
    for two_stage in "${without_two_stage[@]}"; do
      echo "Running experiment with mask_ratio=$mask_ratio, weight=$weight, without_two_stage=$two_stage"
      
      # Run the Python script with the current combination of parameters
      python main.py \
        -dataset HYBRID \
        -lifelong_name MGND \
        -using_MAE_loss True \
        -MAE_loss_weights 0.001 0.001 0.001 0.001 \
        -mask_ratio "$mask_ratio" \
        -multi_layer_weight "$weight" \
        -without_two_stage "$two_stage" 
        # -save_path "${save_dir}mask_${mask_ratio}_weight_${weight}_two_stage_${two_stage}/"

      echo "Experiment completed for mask_ratio=$mask_ratio, weight=$weight, without_two_stage=$two_stage"
    done
  done
done

echo "Grid search completed."