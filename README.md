# MetaFarm - CINECA Leonardo HPC Training

Automated training pipeline for MetaFarm agricultural AI on CINECA Leonardo supercomputer.

## 🚀 Quick Start

```bash
# Clone on CINECA Leonardo
git clone https://github.com/Hmauto/metafarm-cineca.git
cd metafarm-cineca

# Run everything
./master_metafarm.sh
```

## 📁 Files

| File | Purpose |
|------|---------|
| `setup_metafarm_cineca.sh` | Environment setup |
| `master_metafarm.sh` | One-click automation |
| `quickstart.sh` | Quick start wrapper |

## 📚 Documentation

- [METAFARM_STEP_BY_STEP.md](METAFARM_STEP_BY_STEP.md) - Detailed walkthrough
- [METAFARM_CINECA_GUIDE.md](METAFARM_CINECA_GUIDE.md) - Full documentation

## 🎯 CINECA Resources

- **System**: Leonardo (4x A100 GPUs)
- **Partition**: boost_usr_prod
- **Docs**: https://docs.hpc.cineca.it/

## ⚡ Usage

```bash
# Full automation
./master_metafarm.sh

# Test mode (30 min)
./master_metafarm.sh --test-mode

# Step by step
./setup_metafarm_cineca.sh
./scripts/download_datasets.sh
sbatch scripts/run_training.slurm
```

---
Created for MetaFarm agricultural AI training.
