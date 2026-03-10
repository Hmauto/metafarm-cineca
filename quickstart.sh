#!/bin/bash
# ============================================================================
# MetaFarm Quick Start - Copy this to CINECA and run
# ============================================================================
# This is a self-contained quick start script
# Run on CINECA Leonardo: bash quickstart.sh
# ============================================================================

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           MetaFarm Quick Start - CINECA Leonardo              ║"
echo "║     Based on: https://docs.hpc.cineca.it/hpc/leonardo.html    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on CINECA
if [[ ! "$HOSTNAME" =~ leonardo ]]; then
    echo "⚠️  Warning: This script should be run on CINECA Leonardo"
    echo "   Current host: $HOSTNAME"
    echo "   Please SSH to: login.leonardo.cineca.it"
    echo ""
fi

# Create directory structure
echo "📁 Creating directories..."
mkdir -p ~/metafarm/{scripts,data,results}
cd ~/metafarm

# Download the full setup script if not present
if [ ! -f "setup_metafarm_cineca.sh" ]; then
    echo "📥 Downloading setup script..."
    # Replace with your actual URL or use scp to transfer
    echo "Please ensure setup_metafarm_cineca.sh is in ~/metafarm/"
    echo "Run: scp setup_metafarm_cineca.sh username@login.leonardo.cineca.it:~/metafarm/"
    exit 1
fi

# Run setup
echo ""
echo "🔧 Running setup (this takes ~5-10 minutes)..."
bash setup_metafarm_cineca.sh

# Quick test
echo ""
echo "🧪 Running quick test..."
bash scripts/quick_test.sh

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                      Setup Complete!                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Download data:   ./scripts/download_datasets.sh"
echo "  2. Preprocess:      python scripts/preprocess_data.py --source data/raw/... --output data/"
echo "  3. Train:           sbatch scripts/run_training.slurm"
echo "  4. Monitor:         squeue --me"
echo ""
echo "Documentation: https://docs.hpc.cineca.it/hpc/leonardo.html"
echo ""
