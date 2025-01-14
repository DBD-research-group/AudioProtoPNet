#!/bin/bash
#SBATCH --job-name=convnext_XCL_seed45
#SBATCH --output=/mnt/data/rheinrich/DBD/audioprotopnet/job_logs/convnext_XCL_seed45_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=150:00:00
#SBATCH --nodelist=node4.omnia.test

# Print date, hostname, and working directory
date;hostname;pwd

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
cd /mnt/home/rheinrich/deep_bird_detect/audioprotopnet/audioprotopnet/training

# Check available CPUs and GPUs
echo "Available resources:"
sinfo -o "%N %C %G"

echo "Starting experiment: convnext_XCL_seed45 at $(date)"
srun --exclusive python train_benchmarks.py experiment="XCL/convnext/training/convnext" seed=45
if [[ $? -ne 0 ]]; then
    echo "Experiment convnext_XCL_seed45 failed at $(date)"
    exit 1
fi

echo "All experiments completed successfully at $(date)"
