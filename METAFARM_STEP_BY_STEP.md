# MetaFarm on CINECA Leonardo - Step-by-Step Guide
## Complete Walkthrough with CINECA Documentation

---

## 📚 CINECA Documentation References

This guide follows CINECA's official documentation:
- **Leonardo User Guide**: https://docs.hpc.cineca.it/hpc/leonardo.html
- **SLURM Scheduler**: https://docs.hpc.cineca.it/hpc/hpc_scheduler.html
- **GPU Computing**: https://docs.hpc.cineca.it/hpc/gpu.html
- **Deep Learning Profile**: https://docs.hpc.cineca.it/hpc/hpc_software.html

---

## 🔧 Step 1: Connect to CINECA Leonardo

### 1.1 SSH Login
```bash
# Standard SSH
ssh -i ~/.ssh/your_key username@login.leonardo.cineca.it

# Or if using Grid certificate
gsissh -p 2222 username@login.leonardo.cineca.it
```

### 1.2 Verify Access
```bash
# Check who you are
whoami

# Check available storage
df -h $HOME
df -h $SCRATCH
df -h $WORK

# List available modules
module avail
```

---

## 📦 Step 2: Transfer Setup Scripts to CINECA

### 2.1 From Your Local Machine
```bash
# Create a tar archive of the scripts
tar -czvf metafarm_scripts.tar.gz setup_metafarm_cineca.sh master_metafarm.sh

# Transfer to CINECA
scp metafarm_scripts.tar.gz username@login.leonardo.cineca.it:~/

# SSH into CINECA and extract
ssh username@login.leonardo.cineca.it
tar -xzvf metafarm_scripts.tar.gz
```

### 2.2 Alternative: Direct Download
If you have the scripts on GitHub or a URL:
```bash
# On CINECA login node
wget https://your-domain.com/setup_metafarm_cineca.sh
wget https://your-domain.com/master_metafarm.sh
chmod +x *.sh
```

---

## 🚀 Step 3: Run Automated Setup

### 3.1 Full Automated Setup (Recommended)
```bash
# Run the master script - it does EVERYTHING
./master_metafarm.sh

# Or with test mode (quick validation)
./master_metafarm.sh --test-mode

# Or skip setup if already done
./master_metafarm.sh --skip-setup
```

### 3.2 Manual Step-by-Step (If you prefer)
```bash
# Step 1: Setup environment
./setup_metafarm_cineca.sh

# Step 2: Download datasets
cd ~/metafarm
./scripts/download_datasets.sh

# Step 3: Preprocess
python scripts/preprocess_data.py \
    --source data/raw/PlantVillage-Dataset/raw/color \
    --output $SCRATCH/metafarm/data

# Step 4: Submit job
sbatch scripts/run_training.slurm
```

---

## 🎯 Step 4: Monitor Training

### 4.1 Check Job Status
```bash
# See all your jobs
squeue --me

# See specific job
scontrol show job <JOB_ID>

# See cluster status
sinfo
```

### 4.2 View Logs
```bash
# Real-time log monitoring
tail -f metafarm_training_<JOB_ID>.out

# Error logs
tail -f metafarm_training_<JOB_ID>.err

# Both
watch -n 5 'tail -20 metafarm_training_*.out'
```

### 4.3 Resource Monitoring (while job runs)
```bash
# On the compute node (if you have interactive access)
nvidia-smi

# Check GPU utilization
watch -n 1 nvidia-smi

# Check CPU/Memory
top
htop
```

---

## 📊 Step 5: Retrieve Results

### 5.1 Results Location
Training results are saved to:
- **During training**: `$SCRATCH/metafarm/results_<JOB_ID>/`
- **After training**: Copied to `~/metafarm/results/`

### 5.2 Download Results to Local Machine
```bash
# From your local machine
scp -r username@login.leonardo.cineca.it:~/metafarm/results ./metafarm_results

# Or specific model
scp username@login.leonardo.cineca.it:~/metafarm/results/metafarm_best_model.pth ./
```

### 5.3 Results Files
```
results/
├── metafarm_best_model.pth       # Best validation accuracy model
├── metafarm_final_model.pth      # Final epoch model
├── training_history.json         # Loss/accuracy history
├── logs/                         # TensorBoard logs
│   └── events.out.tfevents.*
├── confusion_matrix.png          # (if generated)
└── *.out, *.err                  # SLURM output files
```

---

## 🔬 Step 6: Validate Results

### 6.1 Check Training History
```bash
# On CINECA or locally
cd ~/metafarm/results
cat training_history.json | python -m json.tool
```

### 6.2 TensorBoard Visualization
```bash
# On CINECA (in a new terminal/session)
module load python/3.11.7
source ~/metafarm/venv/bin/activate
tensorboard --logdir=~/metafarm/results/logs --port=6006

# Forward port to local machine (from local terminal)
ssh -L 6006:localhost:6006 username@login.leonardo.cineca.it

# Open in browser: http://localhost:6006
```

---

## ⚙️ CINECA-Specific Configuration

### Storage Structure (from CINECA docs)
| Path | Purpose | Persistence | Quota |
|------|---------|-------------|-------|
| `$HOME` | Scripts, configs | Permanent | ~50 GB |
| `$SCRATCH` | Temporary data | 30 days | ~10 TB |
| `$WORK` | Project data | Permanent* | Varies |

*Check your project allocation

### SLURM Partitions (Leonardo)
| Partition | GPUs | Max Time | Use Case |
|-----------|------|----------|----------|
| `boost_usr_prod` | 4x A100 | 24h | Standard training |
| `boost_qos_dbg` | 4x A100 | 1h | Debugging |

### Module System
```bash
# Load modules (required)
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

# Save as default
module save metafarm

# Restore
module restore metafarm
```

---

## 🐛 Troubleshooting

### Issue 1: Job Pending (PD status)
```bash
# Check why job is pending
squeue --me -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Reasons:
# - Resources: Waiting for GPUs
# - Priority: Higher priority jobs first
# - QOS: Quality of Service limits

# Solution: Wait or use debug queue
sbatch --qos=boost_qos_dbg scripts/run_training.slurm
```

### Issue 2: Out of Memory
```bash
# Error: CUDA out of memory
# Solution 1: Reduce batch size
# Edit scripts/run_training.slurm:
# --batch-size 32 (instead of 64)

# Solution 2: Request more memory
#SBATCH --mem=240GB

# Solution 3: Use gradient accumulation
# Add to train_metafarm.py:
# --accumulation-steps 4
```

### Issue 3: Module Not Found
```bash
# Error: module: command not found
# Solution: You're not on a login node
# SSH to login node:
ssh login.leonardo.cineca.it
```

### Issue 4: Permission Denied
```bash
# Make scripts executable
chmod +x *.sh
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

### Issue 5: Data Download Fails
```bash
# Manual download alternative
cd ~/metafarm/data/raw
wget --no-check-certificate https://github.com/.../plantvillage.zip

# Or use curl
curl -L -o plantvillage.zip https://github.com/.../plantvillage.zip
```

---

## 📈 Performance Optimization

### For Leonardo A100 GPUs
```bash
# Optimal settings for 4x A100 64GB
BATCH_SIZE=128      # Can go higher with 64GB GPUs
NUM_WORKERS=16      # Half of CPU cores
PIN_MEMORY=true     # Faster CPU→GPU transfer
```

### Multi-GPU Training
Already configured in `train_metafarm.py`:
```python
if gpu_count > 1:
    model = nn.DataParallel(model)
```

### Distributed Training (Advanced)
For even faster training across multiple nodes:
```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
# Run with torchrun (already documented in guide)
```

---

## 📋 Quick Reference Card

```bash
# CONNECT
ssh username@login.leonardo.cineca.it

# SETUP (one time)
./master_metafarm.sh

# TRAIN
sbatch scripts/run_training.slurm

# MONITOR
squeue --me
tail -f metafarm_*.out

# CANCEL
scancel <JOB_ID>

# DOWNLOAD RESULTS
scp -r username@login.leonardo.cineca.it:~/metafarm/results .
```

---

## ✅ Pre-Flight Checklist

Before running:
- [ ] SSH access to Leonardo working
- [ ] `module avail` shows available modules
- [ ] `$SCRATCH` directory accessible
- [ ] Setup scripts transferred to CINECA
- [ ] Scripts are executable (`chmod +x`)
- [ ] Sufficient quota in `$HOME` and `$SCRATCH`

---

## 📞 CINECA Support

- **Documentation**: https://docs.hpc.cineca.it/
- **Support Email**: superc@cineca.it
- **Account Issues**: accounts@cineca.it
- **Service Status**: https://status.hpc.cineca.it/

---

## 🎓 Additional Resources

### CINECA Training Materials
- **HPC Courses**: https://www.hpc.cineca.it/content/courses
- **Best Practices**: https://docs.hpc.cineca.it/hpc/best_practices.html

### PyTorch on HPC
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Multi-GPU Guide**: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

---

**Ready to train! Run `./master_metafarm.sh` on CINECA Leonardo** 🚀
