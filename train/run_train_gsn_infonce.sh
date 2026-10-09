#!/bin/bash
#SBATCH --job-name=gsn_nce
#SBATCH --output=gsn_nce_tau%a_%A.out
#SBATCH --error=gsn_nce_tau%a_%A.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=10
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# sGSN-RS (LSTM) with the temperature-scaled semantic loss. Identical to
# run_train_gpn_rs_noavs.sh in EVERY other respect -- same architecture, same AVS exclusion,
# same probe, same tm=1 -- so that run is the paired baseline and the only difference between
# the two families is the shape of the semantic loss:
#
#   --semantic_loss 1   sim[mask==3].mean() - sim[mask==1].mean()   (published, flat mean)
#   --semantic_loss 2   symmetric InfoNCE at temperature tau        (this script)
#
# The published loss is the tau -> inf limit of the InfoNCE one (test_semantic_infonce.py
# measures cos(grad, published grad) = 1.00000 at tau=100), so tau is a continuous knob
# between "every cross-scene negative weighted equally" and "the gradient concentrated on the
# hard, near-confusable scenes". --semantic_norm tau keeps gradient magnitude tau-independent,
# so the sweep below is not also a learning-rate sweep.
#
# Why this is worth a checkpoint family: the processing-demand account of fixation duration
# (Sulewski et al. 2026) rests on recognition difficulty. A sharpened loss forces the
# recurrent state to resolve exactly the near-confusable cases, which is a route to a
# difficulty-sensitive gate that does not pass through the memory account. The read-out is
# the WS1 probe plus the full WS4 evaluation battery (memgate_v2/06_architecture_roadmap.md).
#
# Prerequisites: identical to run_train_gpn_rs_noavs.sh (exclusion indices + probe pack).
#
# BBV selects the glimpse backbone without editing this file: BBV=4 sbatch run_train_gsn_infonce.sh
# runs the same sweep on the DVD-B embeddings. bbv is parameter-identical (both backbones are
# 2048-d, input_feats is unchanged), so only the input statistics differ -- but the probe pack
# must then be the DVD one, because it stores already-embedded glimpses. PACK follows BBV below,
# and the probe refuses to run if the two disagree.

# Load environment
source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

export TMPDIR=/share/klab/psulewski/psulewski/pyavs/tmp
mkdir -p "${TMPDIR}"

# tau sweep. 0.2 is mildly sharpened, 0.05 is aggressive; the baseline run is the tau -> inf
# end of the same axis. Expect the loss floor (contrastive_floor) to rise with sharpening --
# it is set by how confusable the MPNet caption embeddings themselves are at this temperature,
# and that is the quantity the change is meant to put pressure on.
TAUS=(0.2 0.1 0.05)
TAU=${TAUS[${SLURM_ARRAY_TASK_ID}]}

TM=1
BBV=${BBV:-6}
if [ "${BBV}" = "6" ]; then PACK_SUFFIX=""; else PACK_SUFFIX="_dvd"; fi   # bbv 4 = DVD-B
PACK=/share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack${PACK_SUFFIX}_v1.h5
HELDOUT=/share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout

echo "=========================================="
echo "Training sGSN-RS (LSTM) tm=${TM}  InfoNCE semantic loss  tau=${TAU}  bbv=${BBV}"
echo "Job ID: ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID}"
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
    --bbv ${BBV} \
    --gaze_type dg3 \
    --input_dropout 0.25 \
    --rnn_dropout 0.1 \
    --glimpse_loss 0 \
    --semantic_loss 2 \
    --semantic_temp ${TAU} \
    --semantic_norm tau \
    --semantic_dedup 1 \
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

echo "Training complete for sGSN-RS InfoNCE tau=${TAU} bbv=${BBV}"
