#!/bin/bash
#SBATCH --job-name=baseline_xview2_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.out
#SBATCH --error=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.err
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

set -euo pipefail

PROJECT_ROOT=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision
JOBS_ROOT=$PROJECT_ROOT/jobs
LOG_ROOT=$JOBS_ROOT/logs
RUNS_ROOT=$JOBS_ROOT/runs
RUN_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR=$RUNS_ROOT/$RUN_ID
BOOTSTRAP_STDOUT=$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
BOOTSTRAP_STDERR=$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err

mkdir -p "$LOG_ROOT" "$RUNS_ROOT" "$RUN_DIR"

if [ -f "$BOOTSTRAP_STDOUT" ]; then
  mv "$BOOTSTRAP_STDOUT" "$RUN_DIR/slurm.out"
fi
if [ -f "$BOOTSTRAP_STDERR" ]; then
  mv "$BOOTSTRAP_STDERR" "$RUN_DIR/slurm.err"
fi

cd "$JOBS_ROOT"

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

source /sfs/gpfs/tardis/home/xxn2pb/venvs/disaster_vision/bin/activate

cp "$0" "$RUN_DIR/submitted_job.slurm"
cp "$JOBS_ROOT/siamese_baseline_v2.py" "$RUN_DIR/siamese_baseline_v2.py"

echo "=============================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Run directory: $RUN_DIR"
echo "=============================="

{
  echo "run_id=$RUN_ID"
  echo "job_name=$SLURM_JOB_NAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "hostname=$(hostname)"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "working_dir=$(pwd)"
} > "$RUN_DIR/job_info.txt"

echo "Python location:"
which python

echo "Python version:"
python --version

echo "GPU check:"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt" || true

echo "Checking key paths..."
ls "$JOBS_ROOT"
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/tier1
ls /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/hold

start=$(date +%s)

python "$JOBS_ROOT/siamese_baseline_v2.py" \
  --train_root /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/tier1 \
  --val_root /sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/hold \
  --save_dir "$RUN_DIR" \
  --epochs 50 \
  --batch_size 4 \
  --image_size 256 \
  --lr 3e-4 \
  --num_workers 4 \
  --selection_metric joint

end=$(date +%s)

echo "=============================="
echo "Job ended at: $(date)"
echo "Total runtime: $((end - start)) seconds"
echo "=============================="
