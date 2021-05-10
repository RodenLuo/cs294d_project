#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J run_wt103_large_parallel2
#SBATCH -o run_wt103_large_parallel2.%J.out
#SBATCH -e run_wt103_large_parallel2.%J.err
#SBATCH --mail-user=deng.luo@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=4-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --constraint=[v100]

eval "$(conda shell.bash hook)"
conda activate pt1.4
bash run_wt103_large_parallel2.sh train --work_dir parallel2_large_new
