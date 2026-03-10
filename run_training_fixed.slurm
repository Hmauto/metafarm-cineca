#!/bin/bash
#SBATCH --job-name=metafarm_training
#SBATCH --time=04:00:00              # Reduced to 4 hours for aih4a_metafarm account
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2            # Reduced to 2 GPUs (can request 4 if available)
#SBATCH --cpus-per-task=16           # Adjusted for 2 GPUs
#SBATCH --mem=64GB                   # Reduced memory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Account: using default (aih4a_metafarm)

echo "=========================================="
echo "  MetaFarm Training - CINECA Leonardo"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Account: aih4a_metafarm"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load CINECA modules
module purge
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

echo ""
echo "Modules loaded:"
module list

# Setup paths
METAFARM_DIR="$HOME/metafarm"
SCRATCH_DIR="$SCRATCH/metafarm"
DATA_DIR="${SCRATCH_DIR}/data"
RESULTS_DIR="${SCRATCH_DIR}/results_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR

# Activate environment
source $METAFARM_DIR/venv/bin/activate

echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "Starting training..."
echo "=========================================="

# Run training with aih4a_metafarm-optimized settings
python $METAFARM_DIR/scripts/train_metafarm.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-workers 8

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results: $RESULTS_DIR"
echo "Finished: $(date)"
echo "=========================================="

# Copy results to persistent storage
cp -r $RESULTS_DIR $METAFARM_DIR/results/
echo "Results copied to: $METAFARM_DIR/results/"
