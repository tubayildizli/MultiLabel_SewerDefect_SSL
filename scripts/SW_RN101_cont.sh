#!/bin/bash
#SBATCH --job-name="SW_RN101_cont"
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=research-ceg-wm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t.yildizli@tudelft.nl

# Load modules:
module load miniconda3
module load cuda/11.7
module load 2024r1

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

conda activate vissl
srun python tools/run_distributed_engines_main.py \
  hydra.verbose=true \
  config=pretrain/swav/swav_1_gpu_resnet101_cont.yaml \
  config.DATA.TRAIN.DATASET_NAMES=[Sewer_ML] \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATA_PATHS=["${YOUR_PROJECT_ROOT}/data/pretrain"] \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=128 \
  config.OPTIMIZER.num_epochs=200 \
  config.CHECKPOINT.DIR="${YOUR_PROJECT_ROOT}/checkpoints/${WHICH_DATA}/SW_RN101_FTAL/vissl" \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
  

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate