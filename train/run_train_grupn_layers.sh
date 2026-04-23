#!/bin/bash
#SBATCH --job-name=grupn_train
#SBATCH --output=grupn_train_tm%a_%A.out
#SBATCH --error=grupn_train_tm%a_%A.err
#SBATCH --array=1-2
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# Load environment
source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

# SLURM_ARRAY_TASK_ID = number of RNN layers (1 or 2)
TM=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Training GRUPN (GRU)  tm=${TM}"
echo "Job ID: ${SLURM_JOB_ID}  Array task: ${TM}"
echo "Node: ${SLURMD_NODENAME}"
echo "=========================================="

cd /home/student/p/psulewski/GRUPN/train

python train_net.py \
    --network_type gru \
    --timestep_multiplier ${TM} \
    --n_rnn 1024 \
    --timesteps 6 \
    --recurrence 1 \
    --provide_loc 0 \
    --bbv 6 \
    --gaze_type dg3 \
    --input_dropout 0.25 \
    --rnn_dropout 0.1 \
    --glimpse_loss 0 \
    --semantic_loss 1 \
    --scene_loss 0 \
    --gazeloc_loss 0 \
    --input_split 0 \
    --regularisation 1 \
    --trainer train_515 \
    --dva_dataset NSD \
    --learning_rate 0.0001 \
    --network_id 1

echo "Training complete for GRUPN tm=${TM}"
