#!/bin/bash
###############################################################################
# MetaFarm - Complete Automated Setup for CINECA Leonardo HPC
# This script sets up everything needed to train MetaFarm on CINECA
# Based on CINECA official documentation: https://docs.hpc.cineca.it/
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "  MetaFarm CINECA Leonardo Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
METAFARM_DIR="$HOME/metafarm"
DATA_DIR="$METAFARM_DIR/data"
MODELS_DIR="$METAFARM_DIR/models"
SCRIPTS_DIR="$METAFARM_DIR/scripts"
RESULTS_DIR="$METAFARM_DIR/results"

# CINECA specific paths
SCRATCH_DIR="$SCRATCH/metafarm"
WORK_DIR="$WORK/metafarm" 2>/dev/null || WORK_DIR="$HOME/metafarm_work"

echo -e "${YELLOW}Step 1: Creating directory structure...${NC}"
mkdir -p $METAFARM_DIR/{data,models,scripts,results}
mkdir -p $DATA_DIR/{raw,processed,train,val,test}
mkdir -p $SCRATCH_DIR 2>/dev/null || echo "Note: $SCRATCH not available, using $HOME"
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo -e "${YELLOW}Step 2: Loading CINECA modules...${NC}"
module purge
module load python/3.11.7 || module load python/3.10.9 || module load python/3.9.15
module load profile/deeplrn
module load cuda/11.8 || module load cuda/12.1
module list
echo -e "${GREEN}✓ Modules loaded${NC}"

echo ""
echo -e "${YELLOW}Step 3: Creating Python virtual environment...${NC}"
cd $METAFARM_DIR
python3 -m venv venv
source venv/bin/activate

echo -e "${YELLOW}Step 4: Installing Python packages...${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm scikit-learn matplotlib pandas seaborn opencv-python Pillow numpy tqdm tensorboard
pip install onnx onnxruntime

echo -e "${GREEN}✓ Python environment ready${NC}"

echo ""
echo -e "${YELLOW}Step 5: Creating training scripts...${NC}"

cat > $SCRIPTS_DIR/train_metafarm.py << 'PYEOF'
#!/usr/bin/env python3
"""
MetaFarm Model Training on CINECA HPC
Optimized for Leonardo - 4x A100 GPUs
Based on CINECA Deep Learning best practices
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import timm
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# CINECA HPC Configuration
def setup_device():
    """Setup device for CINECA Leonardo"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🖥️  CINECA Leonardo GPU Setup")
        print(f"   GPUs available: {gpu_count}")
        print(f"   Primary GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        return device, gpu_count
    else:
        print("⚠️  No GPU available, using CPU")
        return torch.device('cpu'), 0

def get_data_transforms(image_size=224):
    """Get data transforms for agricultural images"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, model_name='efficientnet_b0', pretrained=True):
    """Create model for MetaFarm"""
    print(f"🔄 Creating {model_name} model...")
    
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        if batch_idx % 50 == 0:
            batch_acc = 100. * correct / total
            print(f"   Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.2f}%")
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    if writer:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc, epoch_time

def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser(description='MetaFarm Training on CINECA')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers')
    
    args = parser.parse_args()
    
    # Setup
    device, gpu_count = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Data transforms
    train_transform, val_transform = get_data_transforms(args.image_size)
    
    # Load datasets
    print(f"\n📂 Loading datasets from {args.data_dir}")
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    num_classes = len(train_dataset.classes)
    print(f"   Classes: {num_classes}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # DataLoaders (optimized for CINECA)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = create_model(num_classes, args.model)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        print("   Backbone frozen (last 20 layers trainable)")
    
    model = model.to(device)
    
    # Multi-GPU if available
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print(f"   Using {gpu_count} GPUs with DataParallel")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'epochs': []
    }
    
    for epoch in range(args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc, preds, targets = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"   Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Time: {train_time:.1f}s")
        print(f"   Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': train_dataset.classes,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'metafarm_best_model.pth'))
            print(f"   💾 Saved best model (Acc: {val_acc:.2f}%)")
    
    # Save final model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), 
               os.path.join(args.output_dir, 'metafarm_final_model.pth'))
    
    # Save history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Best validation accuracy: {best_acc:.2f}%")
    print(f"   Results saved to: {args.output_dir}")
    
    writer.close()

if __name__ == '__main__':
    main()
PYEOF

echo -e "${GREEN}✓ Training script created${NC}"

echo ""
echo -e "${YELLOW}Step 6: Creating SLURM job script...${NC}"

cat > $SCRIPTS_DIR/run_training.slurm << 'SLURMEOF'
#!/bin/bash
#SBATCH --job-name=metafarm_training
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# CINECA Leonardo HPC Configuration
# Documentation: https://docs.hpc.cineca.it/hpc/leonardo.html

echo "=========================================="
echo "  MetaFarm Training - CINECA Leonardo"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

# Run training with CINECA-optimized settings
python $METAFARM_DIR/scripts/train_metafarm.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 16

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results: $RESULTS_DIR"
echo "Finished: $(date)"
echo "=========================================="

# Copy results to persistent storage
cp -r $RESULTS_DIR $METAFARM_DIR/results/
echo "Results copied to: $METAFARM_DIR/results/"
SLURMEOF

echo -e "${GREEN}✓ SLURM job script created${NC}"

echo ""
echo -e "${YELLOW}Step 7: Creating data download script...${NC}"

cat > $SCRIPTS_DIR/download_datasets.sh << 'DLSEOF'
#!/bin/bash
# Download datasets for MetaFarm training

set -e

DATA_DIR="${1:-$HOME/metafarm/data/raw}"
echo "Downloading datasets to: $DATA_DIR"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo ""
echo "=========================================="
echo "  Downloading MetaFarm Datasets"
echo "=========================================="

# 1. PlantVillage Dataset (Primary)
echo ""
echo "📥 Downloading PlantVillage dataset..."
echo "   Source: https://github.com/spMohanty/PlantVillage-Dataset"
if [ ! -d "PlantVillage-Dataset" ]; then
    wget -q --show-progress https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip -O plantvillage.zip
    unzip -q plantvillage.zip
    mv PlantVillage-Dataset-master PlantVillage-Dataset
    rm plantvillage.zip
    echo "   ✓ PlantVillage downloaded"
else
    echo "   ✓ PlantVillage already exists"
fi

# 2. Create directory structure for processed data
echo ""
echo "📁 Creating data structure..."
mkdir -p $DATA_DIR/../{processed,train,val,test}

echo ""
echo "=========================================="
echo "  Dataset Download Complete!"
echo "=========================================="
echo "Raw data: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Process data: ./scripts/preprocess_data.sh"
echo "  2. Submit job: sbatch scripts/run_training.slurm"
echo "=========================================="
DLSEOF

chmod +x $SCRIPTS_DIR/download_datasets.sh

echo -e "${GREEN}✓ Download script created${NC}"

echo ""
echo -e "${YELLOW}Step 8: Creating preprocessing script...${NC}"

cat > $SCRIPTS_DIR/preprocess_data.py << 'PPEOF'
#!/usr/bin/env python3
"""
Preprocess agricultural datasets for MetaFarm
Prepares data for CINECA Leonardo training
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split dataset into train/val/test"""
    random.seed(seed)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process each class
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Get all images
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Split
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copy files
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
        
        print(f"   {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    print("Preprocessing dataset...")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    
    split_dataset(args.source, args.output)
    
    print("\n✅ Preprocessing complete!")

if __name__ == '__main__':
    main()
PPEOF

chmod +x $SCRIPTS_DIR/preprocess_data.py

echo -e "${GREEN}✓ Preprocessing script created${NC}"

echo ""
echo -e "${YELLOW}Step 9: Creating helper scripts...${NC}"

# Monitor script
cat > $SCRIPTS_DIR/monitor.sh << 'MONEOF'
#!/bin/bash
# Monitor MetaFarm training job

if [ -z "$1" ]; then
    echo "Usage: ./monitor.sh <JOB_ID>"
    echo "  or:  ./monitor.sh --status (to see all your jobs)"
    exit 1
fi

if [ "$1" == "--status" ]; then
    echo "Your jobs:"
    squeue --me
    exit 0
fi

JOB_ID=$1
echo "Monitoring job $JOB_ID..."
echo "Press Ctrl+C to stop"
echo ""

tail -f $HOME/metafarm/metafarm_training_${JOB_ID}.out 2>/dev/null || \
tail -f $HOME/metafarm_${JOB_ID}.out 2>/dev/null || \
tail -f metafarm_training_${JOB_ID}.out 2>/dev/null || \
echo "Log file not found. Check with: scontrol show job $JOB_ID"
MONEOF

chmod +x $SCRIPTS_DIR/monitor.sh

# Quick test script
cat > $SCRIPTS_DIR/quick_test.sh << 'QTEOF'
#!/bin/bash
# Quick test on small dataset

echo "Running quick test on CINECA Leonardo..."

METAFARM_DIR="$HOME/metafarm"
source $METAFARM_DIR/venv/bin/activate

# Create small test dataset
TEST_DIR="$METAFARM_DIR/test_data"
mkdir -p $TEST_DIR

# Run training with minimal epochs
python $METAFARM_DIR/scripts/train_metafarm.py \
    --data-dir $TEST_DIR \
    --output-dir $METAFARM_DIR/test_results \
    --epochs 2 \
    --batch-size 8 \
    --num-workers 4

echo "Quick test complete!"
QTEOF

chmod +x $SCRIPTS_DIR/quick_test.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  $METAFARM_DIR/"
echo "  ├── venv/              # Python environment"
echo "  ├── data/              # Datasets"
echo "  │   ├── raw/           # Downloaded data"
echo "  │   ├── train/         # Training split"
echo "  │   ├── val/           # Validation split"
echo "  │   └── test/          # Test split"
echo "  ├── models/            # Saved models"
echo "  ├── scripts/           # Training scripts"
echo "  └── results/           # Training results"
echo ""
echo "Next steps:"
echo "  1. Download datasets:"
echo "     $METAFARM_DIR/scripts/download_datasets.sh"
echo ""
echo "  2. Preprocess data:"
echo "     python $METAFARM_DIR/scripts/preprocess_data.py --source /path/to/raw --output $METAFARM_DIR/data"
echo ""
echo "  3. Submit training job:"
echo "     cd $METAFARM_DIR && sbatch scripts/run_training.slurm"
echo ""
echo "  4. Monitor job:"
echo "     ./scripts/monitor.sh <JOB_ID>"
echo "     or: squeue --me"
echo ""
echo "=========================================="
