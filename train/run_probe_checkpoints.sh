#!/bin/bash
#SBATCH --job-name=probe_ckpts
#SBATCH --output=/share/klab/psulewski/psulewski/memgate/logs/probe_ckpts_%A.out
#SBATCH --error=/share/klab/psulewski/psulewski/memgate/logs/probe_ckpts_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1

# Per-epoch duration-probe betas for a `semc_1` checkpoint family, recovered after training.
#
# Usage:  sbatch run_probe_checkpoints.sh            # bbv 4 (DVD-B)
#         BBV=6 sbatch run_probe_checkpoints.sh      # bbv 6 (SimCLR)
#
# The DVD-B run (job 12918533) was trained with the probe disabled so that training could start
# before the probe pack existed, so it has no live betas. It was trained with
# --save_every_epoch 1, which is what makes them recoverable: the probe is applied to each saved
# epoch and yields the quantity it would have logged live.
#
# BBV=6 is expected to abort with "no ..._epoch_*.pth": the published SimCLR run predates
# --save_every_epoch and kept only its min-val checkpoint. Its per-epoch betas are already
# available from wandb (run kietzmannlab/grupn/p7iuf9a0, exported into
# noavs_tm1_training_history.csv), so nothing needs recomputing on that side.

source ~/.bashrc
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
conda activate lightning

# after the env block: spack/conda init scripts trip -u
set -euo pipefail

export TMPDIR=/share/klab/psulewski/psulewski/memgate/tmp
mkdir -p "${TMPDIR}"

BBV=${BBV:-4}
if [ "${BBV}" = "6" ]; then PACK_SUFFIX=""; LABEL="SimCLR"; else PACK_SUFFIX="_dvd"; LABEL="DVD-B"; fi
PACK=/share/klab/psulewski/psulewski/memdur_paper/data/behav/gpn_features/avs_probe_pack${PACK_SUFFIX}_v1.h5
OUT_DIR=/share/klab/psulewski/psulewski/memgate/probe_curves
mkdir -p "${OUT_DIR}"

NET="gpn_lstm_n_1024_tm_1_t_6_recurrence_1_loc_1_bbv${BBV}_gaze_dg3_indp_0.25_rnndp_0.1_gcpc_0_semc_1_scc_0_locmse_0_insplit_0_reg_1_tr_train_515_noavs_gdva_NSD_lr_0.0001_num_1"
ROOT=/home/student/p/psulewski/GRUPN/train/logs

echo "=========================================="
echo "Post-hoc probe over saved epochs  bbv=${BBV} (${LABEL})"
echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "=========================================="

cd /home/student/p/psulewski/GRUPN/train

python probes/probe_checkpoints.py \
    --net_dir "${ROOT}/net_params/${NET}" \
    --pack "${PACK}" \
    --loss_npz "${ROOT}/perf_logs/${NET}/loss_${NET}.npz" \
    --label "${LABEL}" \
    --layers 0 \
    --stats meta \
    --saccade_units train_units \
    --device cuda \
    --out_csv "${OUT_DIR}/semc1_bbv${BBV}_probe_curve.csv"

echo "Post-hoc probe complete for bbv=${BBV}"
