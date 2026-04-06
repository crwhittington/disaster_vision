#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=standard

cd /sfs/gpfs/tardis/home/xxn2pb/siamese_project
mkdir -p logs

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

echo "Job started at: $(date)"
which python
python --version
time python data/all_data.py
echo "Job ended at: $(date)"