#!/bin/bash
#SBATCH --job-name=audioprotopnet_XCL_inference
#SBATCH --output=/mnt/data/rheinrich/DBD/audioprotopnet/job_logs/audioprotopnet_XCL_inference_%j.log
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
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/ppnet_convnext/2024-09-30_175733/callback_checkpoints/ppnet_convnext_warm.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_audioprotopnet_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_audioprotopnet.py experiment="${experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_1_prototype" ckpt_path="${checkpoint_path}" seed=$seed
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_audioprotopnet_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 43
seed=43
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/ppnet_convnext/2024-09-30_175831/callback_checkpoints/ppnet_convnext_warm.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_audioprotopnet_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_audioprotopnet.py experiment="${experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_1_prototype" ckpt_path="${checkpoint_path}" seed=$seed
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_audioprotopnet_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 44
seed=44
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/ppnet_convnext/2024-09-30_180256/callback_checkpoints/ppnet_convnext_warm.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_audioprotopnet_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_audioprotopnet.py experiment="${experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_1_prototype" ckpt_path="${checkpoint_path}" seed=$seed
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_audioprotopnet_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 45
seed=45
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/ppnet_convnext/2024-09-30_180343/callback_checkpoints/ppnet_convnext_warm.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_audioprotopnet_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_audioprotopnet.py experiment="${experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_1_prototype" ckpt_path="${checkpoint_path}" seed=$seed
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_audioprotopnet_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

# Run experiments for seed 47
seed=47
checkpoint_path="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/ppnet_convnext/2024-10-01_064014/callback_checkpoints/ppnet_convnext_warm.ckpt"
for experiment in "${experiments[@]}"; do
  echo "Starting experiment: ${experiment}_audioprotopnet_convnext_inference_XCL_pretrained_seed${seed} at $(date)"
  srun --exclusive python eval_audioprotopnet.py experiment="${experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_1_prototype" ckpt_path="${checkpoint_path}" seed=$seed
  if [[ $? -ne 0 ]]; then
      echo "Experiment ${experiment}_audioprotopnet_convnext_seed${seed} failed at $(date)"
      exit 1
  fi
done

echo "All experiments completed successfully at $(date)"
