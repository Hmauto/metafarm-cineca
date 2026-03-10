# MetaFarm Training on CINECA HPC - Quick Start Summary

## 🎯 Your Objective
Train MetaFarm's AI model (crop disease detection) on CINECA HPC using GPU resources.

---

## ✅ WHAT YOU NEED

### 1. CINECA HPC Access
- Apply at: https://www.hpc.cineca.it/
- Need: Research project + institutional affiliation
- Once approved: SSH access to Leonardo supercomputer

### 2. Hardware Available (Leonardo)
- **4x NVIDIA A100 GPUs** per node (64GB memory each)
- **32 CPU cores**, 512GB RAM
- **Training time**: 4-8 hours for full model

### 3. Recommended Model
**EfficientNet-B0** - Best for MetaFarm because:
- ✅ Small size (17 MB) - runs on edge devices
- ✅ 99.7% accuracy on crop diseases
- ✅ Fast inference (<10ms)
- ✅ Proven in agricultural applications

### 4. Datasets to Download
| Dataset | Purpose | Size |
|---------|---------|------|
| **PlantVillage** | Main training (38 diseases) | 50,000+ images |
| **Wheat Disease** | Moroccan wheat crops | 5,000+ images |
| **IP102** | Pest detection | 75,000+ images |
| **Your own drone data** | Morocco-specific | Collect locally |

**Download locations:**
```bash
# PlantVillage (most important)
https://github.com/spMohanty/PlantVillage-Dataset

# Or use Kaggle
kaggle datasets download -d emmarex/plantdisease
```

---

## 🚀 QUICK START STEPS

### Step 1: Connect to CINECA
```bash
ssh -i ~/.ssh/your_key username@login.leonardo.cineca.it
```

### Step 2: Setup Environment
```bash
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

pip install --user torch torchvision timm scikit-learn
```

### Step 3: Create Job Script (metafarm.slurm)
```bash
#!/bin/bash
#SBATCH --job-name=metafarm
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --output=metafarm_%j.out

module load python/3.11.7 profile/deeplrn
python train_metafarm.py
```

### Step 4: Submit Job
```bash
sbatch metafarm.slurm
squeue --me  # Check status
```

### Step 5: Download Results
```bash
# After training completes
scp username@login.leonardo.cineca.it:~/metafarm_best_model.pth .
```

---

## 📊 EXPECTED COSTS (CINECA)

| Resource | Consumption | Est. Cost |
|----------|-------------|-----------|
| GPU Hours | 24 hours × 4 GPUs | ~48 GPU-hours |
| CPU Hours | 24 hours × 32 cores | ~768 core-hours |
| Storage | 50 GB | Minimal |

**Note:** Academic access often has free allocation or discounted rates.

---

## 🎓 MODEL TRAINING CODE (Simplified)

```python
import torch
import timm
from torchvision import transforms, datasets

# Setup
DEVICE = torch.device('cuda')
model = timm.create_model('efficientnet_b0', 
                          pretrained=True, 
                          num_classes=38)
model = model.to(DEVICE)

# Data
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('/path/to/train', transform=transforms),
    batch_size=64, shuffle=True
)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Save
torch.save(model.state_dict(), 'metafarm_model.pth')
```

---

## 🔗 ALL FILES GENERATED

| File | Location | Purpose |
|------|----------|---------|
| Complete Guide | `/root/.openclaw/workspace/METAFARM_CINECA_GUIDE.md` | Full documentation |
| Training Script | In guide above | Python training code |
| SLURM Script | In guide above | Job submission |

---

## 📞 NEXT STEPS

1. **Apply for CINECA access** (takes 1-2 weeks)
2. **Download PlantVillage dataset**
3. **Test training on small subset**
4. **Run full training job**
5. **Export model to ONNX** (for mobile deployment)

---

**Questions? Check the full guide: `METAFARM_CINECA_GUIDE.md`**

**Ready to train! 🌾🚀**
