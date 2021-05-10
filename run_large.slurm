#!/bin/bash
#SBATCH -N 1
#SBATCH -J run_wt103_large
#SBATCH -o run_wt103_large.%J.out
#SBATCH -e run_wt103_large.%J.err
#SBATCH --mail-user=deng.luo@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=4-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=[v100]

eval "$(conda shell.bash hook)"
conda activate pt1.4
bash run_wt103_large.sh train --work_dir XL_large
