#!/bin/bash
#SBATCH --job-name=metafarm_training
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm

# CINECA Leonardo HPC - MetaFarm Training
# Cost: 4 GPUs × 4 hours = 16 GPU hours per job

echo "=========================================="
echo "  MetaFarm Training - CINECA Leonardo"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: aih4a_metafarm"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Cost: 4 GPUs × 4 hours = 16 GPU hours"
echo "Started: $(date)"
echo "=========================================="

module purge
module load python/3.11.7 profile/deeplrn cuda/11.8

METAFARM_DIR="$HOME/metafarm"
WORK_DIR="$WORK/metafarm"
SCRATCH_DIR="$SCRATCH/metafarm"
DATA_DIR="${SCRATCH_DIR}/data"
RESULTS_DIR="${SCRATCH_DIR}/results_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $WORK_DIR/models

source $METAFARM_DIR/venv/bin/activate

python $METAFARM_DIR/scripts/train_metafarm.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --num-workers 16

echo ""
echo "Copying results to WORK storage..."
cp -r $RESULTS_DIR $WORK_DIR/models/

echo "Completed: $(date)"
