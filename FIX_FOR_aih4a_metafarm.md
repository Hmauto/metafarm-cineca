# Fix for aih4a_metafarm Account

## Problem
Your account `aih4a_metafarm` has these limits:
- **Max time**: 4 hours (not 24)
- **Max GPUs**: Check with `sacctmgr show qos format=name,maxwall,maxtres%30`

## Solution

Use the fixed script:
```bash
cd ~/metafarm-cineca
sbatch run_training_fixed.slurm
```

## Changes Made

| Setting | Original | Fixed |
|---------|----------|-------|
| Time | 24:00:00 | 04:00:00 |
| GPUs | 4 | 2 |
| CPUs | 32 | 16 |
| Memory | 120GB | 64GB |
| Batch Size | 64 | 32 |

## Check Your Limits

```bash
# Check your account limits
sacctmgr show user $USER format=account,cluster,qos
sacctmgr show qos format=name,maxwall,maxtres%30

# Check partition info
sinfo -p boost_usr_prod -o "%P %a %l %D %N %G %C %m"
```

## Training Time Estimate

With 2x A100 and batch size 32:
- **50 epochs**: ~3-4 hours
- **25 epochs**: ~2 hours

## Alternative: Shorter Run

If you want a quicker test:
```bash
# Edit the script to reduce epochs
sed -i 's/--epochs 50/--epochs 25/' run_training_fixed.slurm
sbatch run_training_fixed.slurm
```
