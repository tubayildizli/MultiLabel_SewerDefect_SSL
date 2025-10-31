#!/bin/bash
#SBATCH --job-name="Sw_RN101_FTAL"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G
#SBATCH --time=48:00:00
#SBATCH --account=research-ceg-wm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t.yildizli@tudelft.nl

# Load modules:
module load miniconda3
module load cuda/11.7

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

conda activate vissl
srun python tools/run_distributed_engines_main.py \
  hydra.verbose=true \
  config=pretrain/swav/swav_1_gpu_resnet101.yaml \
  config.DATA.TRAIN.DATASET_NAMES=[Sewer_ML] \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATA_PATHS=["${YOUR_PROJECT_ROOT}/data/pretrain"] \
  config.OPTIMIZER.num_epochs=80 \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
  config.CHECKPOINT.DIR="${YOUR_PROJECT_ROOT}/checkpoints/SSL_model_weights/SW_RN101_FTAL_80e_b256/vissl" \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
  config.WEIGHTS_INIT.PARAMS_FILE="${YOUR_PROJECT_ROOT}/checkpoints/R-101.pkl" \
  config.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks."\
  config.DISTRIBUTED.NUM_PROC_PER_NODE=2
  

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate