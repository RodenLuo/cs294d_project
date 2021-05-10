#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J run_wt103_parallel_M_dev
#SBATCH -o run_wt103_parallel_M_dev.%J.out
#SBATCH -e run_wt103_parallel_M_dev.%J.err
#SBATCH --mail-user=deng.luo@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --constraint=[v100]

eval "$(conda shell.bash hook)"
conda activate pt1.4
bash run_wt103_parallel_M_dev.sh train --work_dir Parallel_M
