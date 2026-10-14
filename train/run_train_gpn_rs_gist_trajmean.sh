#!/bin/bash
#SBATCH --job-name=gpn_rs_gist_trajmean
#SBATCH --output=/share/klab/psulewski/psulewski/memgate/logs/gpn_rs_gist_trajmean_tm%a_%A.out
#SBATCH --error=/share/klab/psulewski/psulewski/memgate/logs/gpn_rs_gist_trajmean_tm%a_%A.err
#SBATCH --array=1
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=10
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# GPN-RS (LSTM) on DVD-B (bbv 4, standard backbone as of 2026-10-14), with h0/c0 initialised
# from the trajectory-mean-crop gist control (memgate_v2/04_gist_initialisation.md section 4),
# instead of the implicit zero init. Identical to run_train_gpn_rs_dvd.sh in EVERY other
# respect -- same architecture, same AVS exclusion, same semantic loss, same tm -- so that run's
# completed pilot (job 12918533, recovered into semc1_dvd_vs_simclr_history.csv, backbone==
# "DVD-B") is the paired baseline and gist_init is the only difference.
#
# --gist_init 1 --gist_source traj_mean: h0/c0 are projected from mean(actvs_seq, dim=1), i.e.
# the mean of THIS TRACE's own fed crop embeddings. This is a cheap smoke-test/control, NOT the
# real low-SF gist pipeline (steps 1-2 of memgate_v2/04): it leaks every later fixation in the
# trace into the t=0 state by construction, and carries foveal (not peripheral/low-SF) content.
# Any duration-beta gain here must be read as "does *any* non-zero scene-correlated init move
# the betas" and reported as such -- never as evidence for the gist hypothesis itself.
#
# --probe 1, --save_every_epoch 1: logs the AVS gate->duration betas live, every epoch, to
# wandb (verified working, memgate_v2/10_implementation_notes.md) -- switched from the
# probe-disabled + post-hoc-recovery approach of run_train_gpn_rs_dvd.sh's pilot after the user
# asked to see context-gate betas during this run rather than wait for it to finish.
# --save_every_epoch 1 is kept regardless, so matched-epoch re-extraction stays possible after
# the fact if a checkpoint needs re-probing (e.g. with --probe_stats mixedlm at a milestone
# epoch): probes/probe_checkpoints.py --net_dir logs/net_params/<net_name> --pack ${PACK} \
# --loss_npz logs/perf_logs/<net_name>/loss_<net_name>.npz --label "gist (traj-mean)" \
# --out_csv gist_trajmean_history.csv
#
# Prerequisites (both already confirmed present on the cluster, 2026-10-14):
#   - exclusion indices aligned to bbv 4: /share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout/
#   - DVD probe pack: /share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack_dvd_v1.h5
#
# --mem: as run_train_gpn_rs_dvd.sh -- the in-memory train_515_noavs glimpse tensor is ~67 GB in
# float32 plus a ~20 GB read buffer; 160GB OOM'd. Drop to --in_memory 0 if 400GB is tight.

# Load environment
source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

# after the env block: spack/conda init scripts trip -u
set -euo pipefail

# node-local /tmp is small and shared across the node; the probe's per-epoch wandb.Image() PNG
# save filled it once before (job 12917643). Redirect temp files to shared scratch regardless.
export TMPDIR=/share/klab/psulewski/psulewski/memgate/tmp
mkdir -p "${TMPDIR}"

TM=${SLURM_ARRAY_TASK_ID}
HELDOUT=/share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout
PACK=/share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack_dvd_v1.h5

echo "=========================================="
echo "Training GPN-RS (LSTM) tm=${TM}  bbv=4 (DVD-B)  gist_init=1 (traj_mean)  AVS scenes held out"
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
    --gist_init 1 \
    --gist_source traj_mean \
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

echo "Training complete for GPN-RS gist_trajmean tm=${TM}"
