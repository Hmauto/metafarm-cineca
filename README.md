# MetaFarm - CINECA Leonardo HPC Training

Automated training pipeline for MetaFarm agricultural AI on CINECA Leonardo supercomputer.

## 🎯 CINECA Resource Allocation

- **Project**: aih4a_metafarm
- **Budget**: 10,000 GPU hours
- **Monthly**: ~4,000 GPU hours
- **Storage**: 1 TB ($WORK)
- **Project End**: March 31, 2026

## 🚀 Quick Start

```bash
# Clone on CINECA Leonardo
git clone https://github.com/Hmauto/metafarm-cineca.git
cd metafarm-cineca

# Run training (16 GPU hours per job)
sbatch scripts/run_training.slurm
```

## 📁 Files

| File | Purpose | Cost |
|------|---------|------|
| `scripts/run_training.slurm` | Main training job | 16 GPU hours |
| `setup_metafarm_cineca.sh` | Environment setup | - |
| `master_metafarm.sh` | One-click automation | - |

## 💰 Budget Tracking

Each job reports its cost in the output:
```
Cost: 4 GPUs × 4 hours = 16 GPU hours
```

Check remaining budget:
```bash
saldo -b
```

## 📊 Resource Usage

- **GPUs**: 4x NVIDIA A100 (full node)
- **Time**: 4 hours per job
- **Cost**: 16 GPU hours per job
- **Can run**: ~625 jobs within budget

## 📚 Documentation

- [METAFARM_STEP_BY_STEP.md](METAFARM_STEP_BY_STEP.md) - Detailed walkthrough
- [METAFARM_CINECA_GUIDE.md](METAFARM_CINECA_GUIDE.md) - Full documentation
- [CINECA docs](https://docs.hpc.cineca.it/hpc/leonardo.html)

---
