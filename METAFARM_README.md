# MetaFarm on CINECA Leonardo - Complete Package

## 📦 What You Have

| File | Size | Purpose |
|------|------|---------|
| **setup_metafarm_cineca.sh** | 22 KB | Complete environment setup |
| **master_metafarm.sh** | 7 KB | One-click full automation |
| **quickstart.sh** | 2.7 KB | Quick start for CINECA |
| **METAFARM_STEP_BY_STEP.md** | 8 KB | Detailed walkthrough |
| **METAFARM_CINECA_GUIDE.md** | 18 KB | Comprehensive documentation |

---

## 🚀 QUICK START (3 Options)

### Option 1: One-Command Automation (Easiest)
```bash
# On CINECA Leonardo
./master_metafarm.sh
```
This does: setup → download → preprocess → train (all automatic)

### Option 2: Test Mode First (Recommended)
```bash
# Test with synthetic data (30 min)
./master_metafarm.sh --test-mode
```

### Option 3: Step-by-Step
```bash
# 1. Setup
./setup_metafarm_cineca.sh

# 2. Download data
./scripts/download_datasets.sh

# 3. Preprocess
python scripts/preprocess_data.py --source data/raw/... --output data/

# 4. Train
sbatch scripts/run_training.slurm
```

---

## 📤 How to Use on CINECA

### Step 1: Transfer Scripts to CINECA

From your local machine:
```bash
# Create archive
tar -czvf metafarm_cineca.tar.gz \
  setup_metafarm_cineca.sh \
  master_metafarm.sh \
  quickstart.sh \
  METAFARM_STEP_BY_STEP.md

# Transfer to CINECA
scp metafarm_cineca.tar.gz username@login.leonardo.cineca.it:~/

# SSH and extract
ssh username@login.leonardo.cineca.it
tar -xzvf metafarm_cineca.tar.gz
```

### Step 2: Run
```bash
chmod +x *.sh
./master_metafarm.sh
```

---

## 🎯 What the Scripts Do

### setup_metafarm_cineca.sh
Creates everything:
```
~/metafarm/
├── venv/                    # Python environment
│   ├── pytorch (CUDA 11.8)
│   ├── timm (EfficientNet)
│   └── all dependencies
├── scripts/
│   ├── train_metafarm.py    # Main training script
│   ├── run_training.slurm   # SLURM job for CINECA
│   ├── download_datasets.sh # Data downloader
│   ├── preprocess_data.py   # Data preprocessor
│   └── monitor.sh           # Job monitor
├── data/                    # Datasets
├── models/                  # Saved models
└── results/                 # Training outputs
```

### master_metafarm.sh
Orchestrates the entire pipeline:
1. Calls setup (if not skipped)
2. Downloads PlantVillage dataset
3. Preprocesses (train/val/test split)
4. Submits SLURM job to CINECA
5. Returns job ID for monitoring

### SLURM Job Configuration
```bash
# From CINECA docs: boost_usr_prod partition
#SBATCH --partition=boost_usr_prod
#SBATCH --gpus-per-task=4        # 4x A100 64GB
#SBATCH --cpus-per-task=32       # 32 cores
#SBATCH --mem=120GB              # 120 GB RAM
#SBATCH --time=24:00:00          # 24 hours max
```

---

## 📊 Training Configuration

### Model: EfficientNet-B0
- **Why**: Best for edge deployment
- **Size**: 17 MB
- **Accuracy**: 95-99% on crop diseases
- **Source**: `timm` library

### Dataset: PlantVillage
- **Size**: 50,000+ images
- **Classes**: 38 crop diseases
- **Split**: 70% train / 15% val / 15% test
- **Preprocessing**: Resize 224x224, normalize

### Hyperparameters
```python
EPOCHS = 50
BATCH_SIZE = 64      # Can increase to 128 on A100
LEARNING_RATE = 1e-4
OPTIMIZER = AdamW
SCHEDULER = CosineAnnealingWarmRestarts
```

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| **Training Time** | 4-8 hours |
| **Final Accuracy** | 95-99% |
| **Model Size** | 17 MB |
| **GPU Memory** | ~12 GB per GPU |
| **Output** | `metafarm_best_model.pth` |

---

## 🔍 Monitoring

```bash
# Check job status
squeue --me

# View logs (real-time)
tail -f metafarm_*.out

# Resource usage
seff <JOB_ID>

# Cancel job
scancel <JOB_ID>
```

---

## 📚 CINECA Documentation Used

This package follows CINECA's official best practices:

| Resource | URL |
|----------|-----|
| Leonardo Guide | https://docs.hpc.cineca.it/hpc/leonardo.html |
| SLURM Scheduler | https://docs.hpc.cineca.it/hpc/hpc_scheduler.html |
| GPU Computing | https://docs.hpc.cineca.it/hpc/gpu.html |
| Deep Learning | https://docs.hpc.cineca.it/hpc/hpc_software.html |

### Key CINECA Features Used
- ✅ `profile/deeplrn` module for ML
- ✅ `boost_usr_prod` partition for GPU
- ✅ `$SCRATCH` for temporary storage
- ✅ SLURM job arrays for batch processing

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Job pending | Wait or use `--qos=boost_qos_dbg` |
| Out of memory | Reduce `--batch-size` to 32 |
| Module not found | Run on login node, not compute node |
| Download fails | Use `wget --no-check-certificate` |
| Permission denied | Run `chmod +x *.sh` |

---

## ✅ Checklist Before Running

- [ ] SSH access to Leonardo working
- [ ] Scripts transferred to `~/metafarm/`
- [ ] Scripts are executable (`chmod +x`)
- [ ] Sufficient quota: `df -h $HOME` and `df -h $SCRATCH`
- [ ] CINECA modules available: `module avail python`

---

## 📞 Support

- **CINECA Help**: superc@cineca.it
- **Documentation**: https://docs.hpc.cineca.it/
- **MetaFarm Scripts**: See `METAFARM_STEP_BY_STEP.md`

---

## 🎓 Next Steps After Training

1. **Download model**:
   ```bash
   scp username@login.leonardo.cineca.it:~/metafarm/results/metafarm_best_model.pth ./
   ```

2. **Export to mobile**:
   ```bash
   python scripts/export_onnx.py
   ```

3. **Deploy to MetaFarm edge devices**

---

**Ready to train! Transfer scripts to CINECA and run `./master_metafarm.sh`** 🚀
