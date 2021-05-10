#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J run_wt103_M
#SBATCH -o run_wt103_M.%J.out
#SBATCH -e run_wt103_M.%J.err
#SBATCH --mail-user=deng.luo@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --constraint=[v100]

eval "$(conda shell.bash hook)"
conda activate pt1.4
bash run_wt103_M.sh train --work_dir M
