#!/bin/bash
#SBATCH --job-name=gpn_rs_dvd
#SBATCH --output=/share/klab/psulewski/psulewski/memgate/logs/gpn_rs_dvd_tm%a_%A.out
#SBATCH --error=/share/klab/psulewski/psulewski/memgate/logs/gpn_rs_dvd_tm%a_%A.err
#SBATCH --array=1
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=10
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# GPN-RS (LSTM) on the DVD-B glimpse embeddings. Identical to run_train_gpn_rs_noavs.sh in
# EVERY other respect -- same architecture, same AVS exclusion, same semantic loss, same probe,
# same tm -- so that run is the paired baseline and the backbone is the only difference:
#
#   --bbv 6   SimCLR ResNet50 1x glimpse embeddings   (published)
#   --bbv 4   DVD-B ResNet50 glimpse embeddings       (this script)
#
# --bbv only selects which r50v{bbv} glimpse dataset CocoGaze opens; both backbones emit 2048-d
# avgpool features, so input_feats and therefore the whole network are parameter-identical. No
# code change is needed to train the swap -- only the matching read-out (encoders.py).
#
# Why a shape-biased backbone is worth a checkpoint: the glimpse stream carries essentially all
# of the published gate->duration effect (i_0_ff +0.104 vs i_0_ctx -0.029 at the saved
# checkpoint, memgate_v2/13). DVD-B's developmental visual diet yields more shape-based, less
# texture-driven representations, so it changes exactly the input the effect rides on while
# leaving the recurrence untouched. The read-out is the WS1 probe plus the WS4 battery
# (memgate_v2/06_architecture_roadmap.md).
#
# --save_every_epoch 1: the probe betas drift strongly over a run's own epochs
# (memgate_v2/13_infonce_sweep_result.md), so a DVD-vs-SimCLR comparison is only valid at a
# matched epoch or at the saved checkpoint. Keeping every epoch's weights makes the
# matched-epoch extraction possible after the fact. ~38 MB x ~15 epochs per run.
#
# Prerequisites (all one-off, in this order):
#   1. the exclusion indices, verified to transfer to the r50v4 files:
#        python -m helpers.scene_overlap --exclusion_dir ${HELDOUT} --check_alignment --bbv 4
#   2. the DVD probe pack (the SimCLR pack cannot be reused -- it stores embedded glimpses):
#        sbatch ../../avs-gazetime/avs_gazetime/memgate/run_build_avs_probe_pack.sh dvd
#
# --mem: as run_train_gpn_rs_noavs.sh -- the in-memory train_515_noavs glimpse tensor is ~67 GB
# in float32 plus a ~20 GB read buffer; 160GB OOM'd. Drop to --in_memory 0 if 400GB is tight.

# Load environment
source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

# after the env block: spack/conda init scripts trip -u
set -euo pipefail

# node-local /tmp is small and shared across the node; the probe's per-epoch wandb.Image() PNG
# save filled it (job 12917643). Redirect temp files to shared scratch.
export TMPDIR=/share/klab/psulewski/psulewski/memgate/tmp
mkdir -p "${TMPDIR}"

TM=${SLURM_ARRAY_TASK_ID}
PACK=/share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack_dvd_v1.h5
HELDOUT=/share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout

echo "=========================================="
echo "Training GPN-RS (LSTM) tm=${TM}  bbv=4 (DVD-B)  AVS scenes held out  + duration probe"
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
    --bbv 4 \
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
    --save_every_epoch 1 \
    --probe 1 \
    --probe_pack ${PACK} \
    --probe_every 1 \
    --probe_stats meta \
    --probe_layers 0 \
    --probe_saccade_units train_units

echo "Training complete for GPN-RS DVD-B tm=${TM}"
