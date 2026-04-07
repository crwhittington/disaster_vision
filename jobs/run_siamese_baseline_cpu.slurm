#!/bin/bash
#SBATCH --job-name=baseline_xview2
#SBATCH --partition=standard
#SBATCH --output=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.out
#SBATCH --error=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -e

cd /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs
mkdir -p /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

source /sfs/gpfs/tardis/home/xxn2pb/venvs/disaster_vision/bin/activate

echo "=============================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "=============================="

echo "Python location:"
which python

echo "Python version:"
python --version

echo "GPU check:"
nvidia-smi || true

echo "Checking key paths..."
ls /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/tier1
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/hold

start=$(date +%s)

python /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/siamese_baseline.py \
  --train_root /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/tier1 \
  --val_root /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/hold \
  --save_dir /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/baseline_test_output \
  --epochs 5 \
  --batch_size 4 \
  --image_size 256 \
  --num_workers 6

end=$(date +%s)

echo "=============================="
echo "Job ended at: $(date)"
echo "Total runtime: $((end - start)) seconds"
echo "=============================="