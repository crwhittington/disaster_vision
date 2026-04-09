#!/bin/bash
#SBATCH --job-name=change_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.out
#SBATCH --error=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision/jobs/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

set -euo pipefail

PROJECT_ROOT=/sfs/gpfs/tardis/home/xxn2pb/disaster_vision
JOBS_ROOT=$PROJECT_ROOT/jobs
LOG_ROOT=$JOBS_ROOT/logs
RUNS_ROOT=$JOBS_ROOT/runs
SCRIPT_PATH=$JOBS_ROOT/change_family_experiments.py

EXPERIMENT_NAME=${EXPERIMENT_NAME:-siamese_shared_concat_bottleneck_only}
TRAIN_ROOT=${TRAIN_ROOT:-/sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/tier1}
VAL_ROOT=${VAL_ROOT:-/sfs/gpfs/tardis/home/xxn2pb/siamese_project/data/xview2_jpeg/hold}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-0.001}
BASE_CHANNELS=${BASE_CHANNELS:-32}
CLS_WEIGHT=${CLS_WEIGHT:-1.0}
DEBUG_SAMPLES=${DEBUG_SAMPLES:-0}
SELECTION_METRIC=${SELECTION_METRIC:-joint}

INPUT_MODE=${INPUT_MODE:-siamese}
ENCODER_MODE=${ENCODER_MODE:-shared}
FUSION_STRATEGY=${FUSION_STRATEGY:-concat}
FUSION_STAGE=${FUSION_STAGE:-bottleneck_only}

RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
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
cp "$SCRIPT_PATH" "$RUN_DIR/change_family_experiments.py"

echo "=============================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Run directory: $RUN_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=============================="

{
  echo "run_id=$RUN_ID"
  echo "job_name=$SLURM_JOB_NAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "hostname=$(hostname)"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "working_dir=$(pwd)"
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "input_mode=$INPUT_MODE"
  echo "encoder_mode=$ENCODER_MODE"
  echo "fusion_strategy=$FUSION_STRATEGY"
  echo "fusion_stage=$FUSION_STAGE"
} > "$RUN_DIR/job_info.txt"

echo "Python location:"
which python

echo "Python version:"
python --version

echo "GPU check:"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt" || true

start=$(date +%s)

python "$SCRIPT_PATH" \
  --train_root "$TRAIN_ROOT" \
  --val_root "$VAL_ROOT" \
  --save_dir "$RUNS_ROOT" \
  --run_name "$RUN_ID" \
  --exp_name "$EXPERIMENT_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --base_channels "$BASE_CHANNELS" \
  --cls_weight "$CLS_WEIGHT" \
  --debug_samples "$DEBUG_SAMPLES" \
  --selection_metric "$SELECTION_METRIC" \
  --input_mode "$INPUT_MODE" \
  --encoder_mode "$ENCODER_MODE" \
  --fusion_strategy "$FUSION_STRATEGY" \
  --fusion_stage "$FUSION_STAGE"

end=$(date +%s)

echo "=============================="
echo "Job ended at: $(date)"
echo "Total runtime: $((end - start)) seconds"
echo "=============================="
