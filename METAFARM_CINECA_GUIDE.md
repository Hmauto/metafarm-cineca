# MetaFarm Model Training on CINECA HPC
## Complete Guide for Precision Agriculture AI

---

## 🖥️ CINECA HPC OVERVIEW

### Available Systems

| System | GPUs | GPU Memory | Use Case |
|--------|------|------------|----------|
| **Leonardo** | NVIDIA A100 (4 per node) | 64 GB HBM2e | Large-scale training |
| **Marconi100** | NVIDIA V100 (4 per node) | 32 GB HBM2 | Medium-scale training |
| **Leonardo Booster** | 4x A100 per node | 64 GB | Multi-GPU distributed training |

### Key Specifications (Leonardo)
- **GPU**: NVIDIA A100 64GB SXM4
- **CPU**: Intel Ice Lake (2x 32 cores)
- **Memory**: 512 GB RAM per node
- **Network**: InfiniBand HDR 200 Gb/s

---

## 📋 STEP 1: ACCESS CINECA HPC

### 1.1 Get Access
```bash
# Apply for access at:
# https://www.hpc.cineca.it/ → User Access → Apply

# You need:
# - Research project proposal
# - Institutional affiliation
# - PI (Principal Investigator) approval
```

### 1.2 Connect to HPC
```bash
# SSH into Leonardo login node
ssh -i ~/.ssh/your_key username@login.leonardo.cineca.it

# Or use gsissh (for Grid certificate users)
gsissh -p 2222 username@login.leonardo.cineca.it
```

### 1.3 Environment Setup
```bash
# Load Python module
module load python/3.11.7

# Load deep learning profile
module load profile/deeplrn

# Verify GPU access
nvidia-smi
```

---

## 📦 STEP 2: DATASETS FOR METAFARM

### Recommended Datasets for Moroccan Agriculture

#### 1. **PlantVillage Dataset** (Primary)
```bash
# Download from: https://github.com/spMohanty/PlantVillage-Dataset
# Contains: 50,000+ images of 38 crop disease categories
# Crops: Tomato, potato, corn, apple, grape, etc.

wget https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip
unzip master.zip
```

#### 2. **Wheat Disease Dataset**
```bash
# Kaggle wheat disease detection
# Suitable for Moroccan wheat farming
kaggle datasets download -d jawadali1045/wheat-disease-detection
```

#### 3. **Crop Pest and Disease Detection (CPDD)**
```bash
# GitHub: https://github.com/xpwu95/IP102
# 102 categories of pests and diseases
# 75,000+ images
```

#### 4. **UAV-Based Agriculture Datasets**
```bash
# BaleUAVision - Hay bale detection
# https://doi.org/10.6084/m9.figshare.28558040

# DeepWeeds - Weed detection
# https://github.com/DeepWeeds/DeepWeeds
```

#### 5. **Morocco-Specific Data Collection**
```bash
# Sentinel-2 satellite data via Google Earth Engine
# Collect your own data from:
# - Souss Valley (agricultural region)
# - Tadla Plain (wheat, citrus)
# - Gharb Plain (rice, sugar beet)

# Use drones for field-specific data
# DJI Phantom 4 Multispectral recommended
```

---

## 🎯 STEP 3: MODEL SELECTION

### Recommended Model: **EfficientNet-B0 with Fine-tuning**

**Why EfficientNet-B0?**
- ✅ 16.76 MB (lightweight for edge deployment)
- ✅ 8.85 million FLOPs (fast inference)
- ✅ 99.69% accuracy on crop disease detection (research-proven)
- ✅ Transfer learning from ImageNet
- ✅ Can run on mobile devices

### Alternative Models

| Model | Size | Accuracy | Best For |
|-------|------|----------|----------|
| **EfficientNet-B0** | 16.8 MB | 99.7% | Edge deployment, mobile |
| **EfficientNet-B3** | 48 MB | 99.8% | Higher accuracy, server |
| **MobileNetV2** | 7 MB | 99.4% | Ultra-lightweight, drones |
| **ResNet-50** | 98 MB | 98.5% | Feature extraction |
| **YOLOv8** | 6-22 MB | 95%+ | Real-time detection |

---

## 📝 STEP 4: CREATE SLURM JOB SCRIPT

### 4.1 Training Script (train_metafarm.py)
```python
#!/usr/bin/env python3
"""
MetaFarm Model Training on CINECA HPC
EfficientNet-B0 for Crop Disease Detection
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm  # For EfficientNet
from sklearn.model_selection import train_test_split
import json

# HPC Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hyperparameters
BATCH_SIZE = 64  # Adjust based on GPU memory
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_CLASSES = 38  # Adjust based on your dataset

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def create_model(num_classes):
    """Create EfficientNet-B0 with custom head"""
    model = timm.create_model('efficientnet_b0', 
                              pretrained=True, 
                              num_classes=num_classes)
    
    # Freeze early layers (optional)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    if scheduler:
        scheduler.step()
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(
        '/path/to/train', 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        '/path/to/val', 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_model(NUM_CLASSES).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'metafarm_best_model.pth')
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'metafarm_final_model.pth')
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
```

### 4.2 SLURM Job Script (metafarm_job.slurm)
```bash
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
#SBATCH --output=metafarm_%j.out
#SBATCH --error=metafarm_%j.err
#SBATCH --account=YOUR_ACCOUNT_NAME

# Print job info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load modules
module purge
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

# Activate conda environment (optional)
# source /path/to/your/conda/bin/activate metafarm_env

# Install required packages
pip install --user timm torchvision scikit-learn matplotlib pandas

# Run training
python train_metafarm.py

echo "Job finished at: $(date)"
```

---

## 🚀 STEP 5: SUBMIT AND MONITOR JOB

### 5.1 Submit Job
```bash
# Submit to SLURM
sbatch metafarm_job.slurm

# Check job status
squeue --me

# Check specific job
scontrol show job <JOB_ID>

# View output in real-time
tail -f metafarm_<JOB_ID>.out
```

### 5.2 Monitor Resources
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Check CPU/Memory
htop

# Check job efficiency
seff <JOB_ID>
```

### 5.3 Common SLURM Commands
```bash
# Cancel job
scancel <JOB_ID>

# Hold job
scontrol hold <JOB_ID>

# Release job
scontrol release <JOB_ID>

# Check queue
squeue -u $USER

# Check cluster status
sinfo
```

---

## 🔧 STEP 6: DISTRIBUTED TRAINING (Multi-GPU)

### 6.1 PyTorch Distributed Training Script
```python
#!/usr/bin/env python3
"""
Distributed MetaFarm Training on CINECA
Uses 4x A100 GPUs on Leonardo
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)
    
    # Your model setup here
    model = create_model(NUM_CLASSES).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    # Training loop
    # ... (same as before)
    
    cleanup()

def main():
    world_size = 4  # 4 GPUs
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

### 6.2 Multi-Node Training Script
```bash
#!/bin/bash
#SBATCH --job-name=metafarm_multi
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB

# Run with torchrun
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):29500 \
    train_distributed.py
```

---

## 📊 STEP 7: MODEL EVALUATION & EXPORT

### 7.1 Evaluation Script
```python
#!/usr/bin/env python3
"""Evaluate MetaFarm Model"""

import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_loader):
    # Load model
    model = timm.create_model('efficientnet_b0', num_classes=38)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Metrics
    print(classification_report(all_targets, all_preds))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    evaluate_model('metafarm_best_model.pth', test_loader)
```

### 7.2 Export to ONNX (for edge deployment)
```python
#!/usr/bin/env python3
"""Export MetaFarm model to ONNX format"""

import torch
import torch.onnx

model = timm.create_model('efficientnet_b0', num_classes=38)
checkpoint = torch.load('metafarm_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "metafarm_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to ONNX format")
```

---

## 🔄 STEP 8: TRAINING PIPELINE AUTOMATION

### 8.1 Full Training Pipeline Script
```bash
#!/bin/bash
# metafarm_pipeline.sh

# Configuration
DATA_DIR="/path/to/agriculture_data"
OUTPUT_DIR="/path/to/output"
EPOCHS=50
BATCH_SIZE=64

# Step 1: Data Preprocessing
echo "Step 1: Preprocessing data..."
python preprocess_data.py \
    --input_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/processed \
    --image_size 224

# Step 2: Data Augmentation
echo "Step 2: Augmenting data..."
python augment_data.py \
    --data_dir $OUTPUT_DIR/processed \
    --augment_factor 3

# Step 3: Train Model
echo "Step 3: Training model..."
sbatch metafarm_job.slurm

# Step 4: Wait for training
sleep 300  # Wait 5 minutes
echo "Training submitted. Monitor with: squeue --me"

# Step 5: Evaluate
echo "Step 5: Will evaluate after training completes"
echo "Run: python evaluate_model.py --model metafarm_best_model.pth"
```

---

## 📈 EXPECTED RESULTS

### Performance Metrics

| Metric | Expected Value |
|--------|----------------|
| **Training Time** | 4-8 hours (4x A100, 50 epochs) |
| **Inference Time** | <10ms per image (GPU) |
| **Model Size** | ~17 MB (EfficientNet-B0) |
| **Accuracy** | 95-99% (crop disease) |
| **F1 Score** | >0.95 |

### Resource Usage

| Resource | Usage |
|----------|-------|
| **GPU Memory** | ~8-12 GB per GPU |
| **CPU Cores** | 32 cores |
| **RAM** | ~60-100 GB |
| **Storage** | ~50 GB (dataset + models) |

---

## 🐛 TROUBLESHOOTING

### Common Issues

```bash
# Issue 1: Out of Memory
# Solution: Reduce batch size
# Edit train_metafarm.py: BATCH_SIZE = 32 (instead of 64)

# Issue 2: CUDA out of memory
# Solution: Use gradient accumulation
python train_metafarm.py --accumulation_steps 4

# Issue 3: Slow data loading
# Solution: Increase num_workers
DataLoader(..., num_workers=16, pin_memory=True)

# Issue 4: Job killed
# Solution: Request more memory
#SBATCH --mem=240GB

# Issue 5: NCCL errors (multi-GPU)
# Solution: Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

---

## 📚 ADDITIONAL RESOURCES

### CINECA Documentation
- **User Guide**: https://docs.hpc.cineca.it/
- **SLURM Guide**: https://docs.hpc.cineca.it/hpc/hpc_scheduler.html
- **GPU Computing**: https://docs.hpc.cineca.it/hpc/gpu.html

### Datasets
- **PlantVillage**: https://github.com/spMohanty/PlantVillage-Dataset
- **AI4AG**: https://www.ai4ag.org/
- **Kaggle Agriculture**: https://www.kaggle.com/datasets?search=agriculture

### Models
- **timm (PyTorch Image Models)**: https://github.com/huggingface/pytorch-image-models
- **EfficientNet**: https://arxiv.org/abs/1905.11946
- **TorchVision**: https://pytorch.org/vision/stable/index.html

---

## ✅ CHECKLIST

- [ ] Apply for CINECA access
- [ ] Set up SSH keys
- [ ] Download datasets
- [ ] Create SLURM job script
- [ ] Test on small dataset first
- [ ] Submit full training job
- [ ] Monitor with `squeue --me`
- [ ] Evaluate model performance
- [ ] Export to ONNX format
- [ ] Download model from HPC

---

**Ready to train MetaFarm on CINECA HPC! 🌾🤖**
