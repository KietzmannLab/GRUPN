#!/bin/bash
#SBATCH --job-name=gpn_rs_noavs
#SBATCH --output=gpn_rs_noavs_tm%a_%A.out
#SBATCH --error=gpn_rs_noavs_tm%a_%A.err
#SBATCH --array=1
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=10
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# GPN-RS (LSTM), identical to run_train_gpn_rs_layers.sh except:
#   --exclude_avs 1  holds all 4080 AVS-MEG stimulus scenes out of training AND validation
#                    (3565 of them were training items under plain train_515 -> leakage control);
#                    the trainer becomes train_515_noavs, so net_name/checkpoints do not collide
#   --probe 1        logs the AVS gate->fixation-duration betas every epoch as a DIAGNOSTIC.
#                    It never feeds back into training: selection stays on the validation loss.
#                    --probe_saccade_units train_units feeds AVS saccades in the units this
#                    network is trained on (fraction of image extent x 256 px, y downward).
#                    The old 'legacy' convention was ~270x too small and left 53% of
#                    coord_proj's units without across-fixation variance.
#
# Prerequisites (both one-off):
#   python -m helpers.scene_overlap --exclusion_dir /share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout
#   sbatch ../../avs-gazetime/avs_gazetime/memgate/run_build_avs_probe_pack.sh
#
# --mem: the in-memory train_515_noavs glimpse tensor is ~67 GB in float32, plus a ~20 GB
# read buffer while the test segment is loaded. Raised 160GB -> 400GB (matching
# run_train_gpn_rs_layers.sh) after the first attempt (job 12917636) OOM'd allocating the
# 67 GB tensor outright at 160GB -- sacct reported COMPLETED 0:0 despite the crash (no set -e).
# Drop to --in_memory 0 if 400GB is still tight.

# Load environment
source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

# node-local /tmp is small and shared across every job on the node; the probe's per-epoch
# wandb.Image() PNG save hit it directly (job 12917643: OSError, No space left on device,
# after epoch 1 completed cleanly). Redirect temp files to shared scratch instead.
export TMPDIR=/share/klab/psulewski/psulewski/pyavs/tmp
mkdir -p "${TMPDIR}"

TM=${SLURM_ARRAY_TASK_ID}
PACK=/share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack_v1.h5
HELDOUT=/share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout

echo "=========================================="
echo "Training GPN-RS (LSTM) tm=${TM}  AVS scenes held out  + duration probe"
echo "Job ID: ${SLURM_JOB_ID}  Array task: ${TM}"
echo "Node: ${SLURMD_NODENAME}"
echo "=========================================="

cd /home/student/p/psulewski/GRUPN/train

python train_net.py \
    --network_type lstm \
    --timestep_multiplier ${TM} \
    --n_rnn 1024 \
    --timesteps 6 \
    --recurrence 1 \
    --provide_loc 1 \
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
    --exclude_avs 1 \
    --heldout_dir ${HELDOUT} \
    --dva_dataset NSD \
    --learning_rate 0.0001 \
    --network_id 1 \
    --probe 1 \
    --probe_pack ${PACK} \
    --probe_every 1 \
    --probe_stats meta \
    --probe_layers 0 \
    --probe_saccade_units train_units

echo "Training complete for GPN-RS noavs tm=${TM}"
