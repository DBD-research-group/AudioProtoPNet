#!/bin/bash
#SBATCH --job-name=convnext_XCL_inference
#SBATCH --output=/mnt/data/rheinrich/DBD/audioprotopnet/job_logs/convnext_XCL_inference_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=150:00:00

# Print date, hostname, and working directory
date; hostname; pwd

# Load Conda environment
echo "Initializing Conda..."
source /etc/profile.d/conda.sh
conda activate audioprotopnet-env

# Check if Conda environment was activated successfully
if [[ $? -ne 0 ]]; then
    echo "Failed to activate Conda environment"
    exit 1
fi

# Select working directory
cd /mnt/home/rheinrich/deep_bird_detect/audioprotopnet/audioprotopnet/evaluation

# Check available CPUs and GPUs
echo "Available resources:"
sinfo -o "%N %C %G"

# Define experiments
experiments=("POW" "PER" "NES" "UHH" "HSN" "NBP" "SSW" "SNE")

# Run experiments for seed 42
seed=42
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-13_202405/callback_checkpoints/convnext_XCL_05.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_benchmarks.py experiment="${experiment}/convnext/inference/convnext_inference_XCL_pretrained" seed=$seed module.network.model.local_checkpoint="${checkpoint_path}"
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 43
seed=43
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-13_202428/callback_checkpoints/convnext_XCL_03.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_benchmarks.py experiment="${experiment}/convnext/inference/convnext_inference_XCL_pretrained" seed=$seed module.network.model.local_checkpoint="${checkpoint_path}"
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 44
seed=44
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-13_202453/callback_checkpoints/convnext_XCL_04.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_benchmarks.py experiment="${experiment}/convnext/inference/convnext_inference_XCL_pretrained" seed=$seed module.network.model.local_checkpoint="${checkpoint_path}"
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 45
seed=45
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202146/callback_checkpoints/convnext_XCL_04.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_benchmarks.py experiment="${experiment}/convnext/inference/convnext_inference_XCL_pretrained" seed=$seed module.network.model.local_checkpoint="${checkpoint_path}"
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 47
seed=47
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202256/callback_checkpoints/convnext_XCL_01.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_benchmarks.py experiment="${experiment}/convnext/inference/convnext_inference_XCL_pretrained" seed=$seed module.network.model.local_checkpoint="${checkpoint_path}"
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

echo "All experiments completed successfully at $(date)"
