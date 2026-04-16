#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu
#SBATCH -J "ML3_SegFormer"
#SBATCH --mail-user=vuk8az@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=ds6050_sp26
#SBATCH --output=segformer_%j.out
#SBATCH --error=segformer_%j.err

cd "/home/vuk8az/DS 6050 Project"

module purge
module load miniforge

source .venv/bin/activate

mkdir -p outputs

python segformer_hpc.py \
  --data_root "/home/vuk8az/DS 6050 Project" \
  --output_dir "/home/vuk8az/DS 6050 Project/outputs" \
  --epochs 100 \
  --batch_size 2 \
  --num_workers 8 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --patience 10