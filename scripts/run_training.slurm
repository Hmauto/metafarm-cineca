#!/bin/bash
#SBATCH --job-name=metafarm_training
#SBATCH --time=04:00:00              # 4 hours = fits monthly budget
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4            # 4x A100 = 32 core hours/hour
#SBATCH --cpus-per-task=32           # Full CPU utilization
#SBATCH --mem=480GB                  # Efficient memory usage
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm     # Your project account

# CINECA Leonardo HPC - MetaFarm Training
# Budget: 10,000 GPU hours | Monthly: ~4,000 GPU hours
# Cost: 4 GPUs × 4 hours = 16 GPU hours per job
# You can run ~625 jobs of this type within budget

echo "=========================================="
echo "  MetaFarm Training - CINECA Leonardo"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Cost: 4 GPUs × 4 hours = 16 GPU hours"
echo "Started: $(date)"
echo "=========================================="

# Load CINECA modules (optimized for deep learning)
module purge
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

echo ""
echo "Modules loaded:"
module list

# Setup paths (use SCRATCH for temp, WORK for results)
METAFARM_DIR="$HOME/metafarm"
WORK_DIR="$WORK/metafarm"
SCRATCH_DIR="$SCRATCH/metafarm"
DATA_DIR="${SCRATCH_DIR}/data"
RESULTS_DIR="${SCRATCH_DIR}/results_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $WORK_DIR/models

# Activate environment
source $METAFARM_DIR/venv/bin/activate

echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "Starting training..."
echo "=========================================="

# Run training with CINECA-optimized settings
python $METAFARM_DIR/scripts/train_metafarm.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --num-workers 16

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results: $RESULTS_DIR"
echo "Finished: $(date)"
echo "=========================================="

# Copy results to persistent WORK storage
cp -r $RESULTS_DIR $WORK_DIR/models/
echo "Results copied to: $WORK_DIR/models/"
