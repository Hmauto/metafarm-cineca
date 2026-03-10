#!/bin/bash
###############################################################################
# MetaFarm - Master Automation Script for CINECA Leonardo
# This script runs the ENTIRE pipeline: setup → download → preprocess → train
# Usage: ./master_metafarm.sh [--skip-setup] [--test-mode]
###############################################################################

set -e

# Parse arguments
SKIP_SETUP=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-setup] [--test-mode]"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

METAFARM_DIR="$HOME/metafarm"
SCRATCH_DIR="$SCRATCH/metafarm" 2>/dev/null || SCRATCH_DIR="$METAFARM_DIR/scratch"

echo "=========================================="
echo -e "${BLUE}  MetaFarm Master Automation Script${NC}"
echo "  CINECA Leonardo HPC"
echo "=========================================="
echo ""
echo "Mode: $([ "$TEST_MODE" = true ] && echo "TEST" || echo "FULL TRAINING")"
echo "Skip setup: $SKIP_SETUP"
echo ""

# ============================================================================
# STEP 1: SETUP
# ============================================================================
if [ "$SKIP_SETUP" = false ]; then
    echo -e "${YELLOW}[1/5] Running setup...${NC}"
    
    if [ -f "setup_metafarm_cineca.sh" ]; then
        bash setup_metafarm_cineca.sh
    else
        echo -e "${RED}Error: setup_metafarm_cineca.sh not found${NC}"
        echo "Please run from the directory containing the setup script"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Setup complete${NC}"
else
    echo -e "${YELLOW}[1/5] Skipping setup (--skip-setup)${NC}"
fi

# ============================================================================
# STEP 2: DOWNLOAD DATASETS
# ============================================================================
echo ""
echo -e "${YELLOW}[2/5] Downloading datasets...${NC}"

if [ "$TEST_MODE" = true ]; then
    echo "Test mode: Creating synthetic data..."
    
    # Create minimal test dataset
    TEST_DATA_DIR="$SCRATCH_DIR/data"
    mkdir -p $TEST_DATA_DIR/{train,val}/test_class
    
    # Generate 10 dummy images for testing
    python3 << 'PYEOF'
from PIL import Image
import os

def create_dummy_image(path, size=(224, 224)):
    img = Image.new('RGB', size, color=(73, 109, 137))
    img.save(path)

train_dir = os.path.expanduser("~/metafarm/scratch/data/train/test_class")
val_dir = os.path.expanduser("~/metafarm/scratch/data/val/test_class")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for i in range(10):
    create_dummy_image(f"{train_dir}/img_{i}.jpg")
    
for i in range(5):
    create_dummy_image(f"{val_dir}/img_{i}.jpg")

print("Created test dataset with 10 train, 5 val images")
PYEOF

else
    # Full dataset download
    echo "Full mode: Downloading PlantVillage..."
    
    cd $METAFARM_DIR/data/raw
    
    # Download PlantVillage if not exists
    if [ ! -d "PlantVillage-Dataset" ]; then
        echo "Downloading PlantVillage dataset (this may take a while)..."
        wget -q --show-progress https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip -O plantvillage.zip || {
            echo -e "${RED}Failed to download PlantVillage${NC}"
            echo "Please download manually from:"
            echo "https://github.com/spMohanty/PlantVillage-Dataset"
            exit 1
        }
        unzip -q plantvillage.zip
        mv PlantVillage-Dataset-master PlantVillage-Dataset
        rm plantvillage.zip
        echo -e "${GREEN}✓ PlantVillage downloaded${NC}"
    else
        echo -e "${GREEN}✓ PlantVillage already exists${NC}"
    fi
fi

# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================
echo ""
echo -e "${YELLOW}[3/5] Preprocessing data...${NC}"

source $METAFARM_DIR/venv/bin/activate

if [ "$TEST_MODE" = true ]; then
    echo "Test mode: Using synthetic data directly"
    # Already structured in step 2
else
    # Preprocess PlantVillage
    RAW_DIR="$METAFARM_DIR/data/raw/PlantVillage-Dataset/raw/color"
    OUTPUT_DIR="$SCRATCH_DIR/data"
    
    if [ -d "$RAW_DIR" ]; then
        echo "Processing PlantVillage dataset..."
        python $METAFARM_DIR/scripts/preprocess_data.py \
            --source $RAW_DIR \
            --output $OUTPUT_DIR
        echo -e "${GREEN}✓ Data preprocessed${NC}"
    else
        echo -e "${YELLOW}Warning: Raw data not found at $RAW_DIR${NC}"
        echo "Please ensure data is downloaded correctly"
    fi
fi

# ============================================================================
# STEP 4: SUBMIT TRAINING JOB
# ============================================================================
echo ""
echo -e "${YELLOW}[4/5] Submitting training job...${NC}"

cd $METAFARM_DIR

if [ "$TEST_MODE" = true ]; then
    # Create test SLURM job
    cat > scripts/run_test.slurm << 'SLURMEOF'
#!/bin/bash
#SBATCH --job-name=metafarm_test
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=test_%j.out

echo "Running MetaFarm test..."
module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/metafarm/venv/bin/activate

python $HOME/metafarm/scripts/train_metafarm.py \
    --data-dir $HOME/metafarm/scratch/data \
    --output-dir $HOME/metafarm/results_test \
    --epochs 2 \
    --batch-size 4 \
    --num-workers 4

echo "Test complete!"
SLURMEOF

    JOB_ID=$(sbatch scripts/run_test.slurm | awk '{print $4}')
    echo -e "${GREEN}✓ Test job submitted: $JOB_ID${NC}"
    
else
    # Submit full training
    if [ -f "scripts/run_training.slurm" ]; then
        # Update data directory in SLURM script
        sed -i "s|DATA_DIR=.*|DATA_DIR=\"$SCRATCH_DIR/data\"|" scripts/run_training.slurm
        
        JOB_ID=$(sbatch scripts/run_training.slurm | awk '{print $4}')
        echo -e "${GREEN}✓ Training job submitted: $JOB_ID${NC}"
    else
        echo -e "${RED}Error: run_training.slurm not found${NC}"
        exit 1
    fi
fi

# ============================================================================
# STEP 5: MONITORING INSTRUCTIONS
# ============================================================================
echo ""
echo "=========================================="
echo -e "${GREEN}[5/5] Pipeline submitted successfully!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Job ID: $JOB_ID${NC}"
echo ""
echo "Monitoring commands:"
echo "  • Check status:   squeue --me"
echo "  • View details:   scontrol show job $JOB_ID"
echo "  • Watch logs:     tail -f metafarm_${JOB_ID}.out"
echo "  • Cancel job:     scancel $JOB_ID"
echo ""
echo "Results will be saved to:"
echo "  $METAFARM_DIR/results/"
echo ""

if [ "$TEST_MODE" = false ]; then
    echo "Expected training time: 4-8 hours"
    echo "You will receive results when complete."
fi

echo ""
echo "=========================================="
