#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J run_text8_base
#SBATCH -o run_text8_base.%J.out
#SBATCH -e run_text8_base.%J.err
#SBATCH --mail-user=deng.luo@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --constraint=[v100]

eval "$(conda shell.bash hook)"
conda activate pt1.4
bash run_text8_base.sh train --work_dir TRAIN
