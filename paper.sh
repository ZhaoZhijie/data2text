#!/bin/bash
#SBATCH --job-name=nlg-model
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 11G # memory pool for all cores
#SBATCH -o nlg-model-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=uoa-gpu
#SBATCH --mail-user=385434432@qq.com

cd /uoa/home/t03zz19/data-to-text-hierarchical

source /uoa/home/t03zz19/zzjenv/bin/activate

python train.py --config train.cfg              
