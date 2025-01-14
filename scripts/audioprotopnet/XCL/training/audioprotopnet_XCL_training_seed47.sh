#!/bin/bash
#SBATCH --job-name=audioprotopnet_XCL_seed47
#SBATCH --output=/mnt/data/rheinrich/DBD/audioprotopnet/job_logs/audioprotopnet_XCL_seed47_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=300:00:00
#SBATCH --nodelist=node3.omnia.test

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

echo "Starting experiment: audioprotopnet_convnext_XCL_seed47 at $(date)"
srun --exclusive python train_audioprotopnet.py experiment="XCL/audioprotopnet/training/audioprotopnet_convnext_1_prototype" seed=47 module.network.model.backbone_model.local_checkpoint="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202256/callback_checkpoints/convnext_XCL_01.ckpt"
if [[ $? -ne 0 ]]; then
    echo "Experiment audioprotopnet_convnext_XCL_seed47 failed at $(date)"
    exit 1
fi

#echo "Starting experiment: audioprotopnet_convnext_XCL_seed47 at $(date)"
#srun --exclusive python train_audioprotopnet.py experiment="XCL/audioprotopnet/training/audioprotopnet_convnext_5_prototypes" seed=47 module.network.model.backbone_model.local_checkpoint="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202256/callback_checkpoints/convnext_XCL_01.ckpt"
#if [[ $? -ne 0 ]]; then
#    echo "Experiment audioprotopnet_convnext_XCL_seed47 failed at $(date)"
#    exit 1
#fi
#
#echo "Starting experiment: audioprotopnet_convnext_XCL_seed47 at $(date)"
#srun --exclusive python train_audioprotopnet.py experiment="XCL/audioprotopnet/training/audioprotopnet_convnext_10_prototypes" seed=47 module.network.model.backbone_model.local_checkpoint="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202256/callback_checkpoints/convnext_XCL_01.ckpt"
#if [[ $? -ne 0 ]]; then
#    echo "Experiment audioprotopnet_convnext_XCL_seed47 failed at $(date)"
#    exit 1
#fi
#
#echo "Starting experiment: audioprotopnet_convnext_XCL_seed47 at $(date)"
#srun --exclusive python train_audioprotopnet.py experiment="XCL/audioprotopnet/training/audioprotopnet_convnext_20_prototypes" seed=47 module.network.model.backbone_model.local_checkpoint="/mnt/oekofor/data/rheinrich/DBD/audioprotopnet/logs/train/runs/XCL/convnext/2024-08-22_202256/callback_checkpoints/convnext_XCL_01.ckpt"
#if [[ $? -ne 0 ]]; then
#    echo "Experiment audioprotopnet_convnext_XCL_seed47 failed at $(date)"
#    exit 1
#fi

echo "All experiments completed successfully at $(date)"
