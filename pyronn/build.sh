#!/bin/bash -l

### current working directory
#SBATCH --chdir=/mnt/home/sun/pyro-nn-torch/

#SBATCH --partition=gpu_A5000
#SBATCH --job-name=build
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/home/sun/pyro-nn-torch/slurm/%j_%x_%Y%m%d.out
#SBATCH --error=/mnt/home/sun/pyro-nn-torch/slurm/%j_%x_%Y%m%d_error.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yipeng.sun@iis-extern.fraunhofer.de

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate pyronn

# run with the specific Python interpreter from the pyronn environment
srun /mnt/home/sun/.conda/envs/pyronn/bin/python -m build
