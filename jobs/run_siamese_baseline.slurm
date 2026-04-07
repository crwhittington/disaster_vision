#!/bin/bash
#SBATCH --job-name=baseline_xview2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G

set -e

cd /sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs
mkdir -p logs

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

source ~/venvs/disaster_vision/bin/activate

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
ls .
ls ../../siamese_project/data/xview2_jpg
ls ../../siamese_project/data/xview2_jpg/tier1
ls ../../siamese_project/data/xview2_jpg/hold

start=$(date +%s)

python siamese_baseline.py \
  --train_root ../../siamese_project/data/xview2_jpg/tier1 \
  --val_root ../../siamese_project/data/xview2_jpg/hold \
  --save_dir baseline_test_output \
  --epochs 2 \
  --batch_size 4 \
  --image_size 256 \
  --num_workers 4

end=$(date +%s)

echo "=============================="
echo "Job ended at: $(date)"
echo "Total runtime: $((end - start)) seconds"
echo "=============================="