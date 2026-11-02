#!/bin/bash
#SBATCH --partition=klab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=randcap_baseline
#SBATCH --output=/share/klab/psulewski/psulewski/memgate/logs/randcap_baseline_%j.out
#SBATCH --error=/share/klab/psulewski/psulewski/memgate/logs/randcap_baseline_%j.err

set -euo pipefail
export TMPDIR=/share/klab/psulewski/psulewski/memgate/tmp

source /etc/profile.d/spack.sh
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate lightning

cd /home/student/p/psulewski/GRUPN/train
python3 random_caption_baseline.py
