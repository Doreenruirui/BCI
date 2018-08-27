#!/bin/bash
#SBATCH --job-name=l495bi
#SBATCH --output=/home/dong.r/BCI/script/train_new/out.s2s.sub.random.fix.49.5.bi
#SBATCH --error=/home/dong.r/BCI/script/train_new/err.s2s.sub.random.fix.49.5.bi
#SBATCH --exclusive
#SBATCH --partition=par-gpu
#SBATCH -N 1

work=/home/dong.r/BCI
cd $work
source ~/.bash_profile
python train.py \
    --data_dir="/gss_gpfs_scratch/dong.r/Dataset/BCI/Sublex/1/1/train/sub" \
    --dev="test" \
    --train_dir="/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi" \
    --size=512 \
    --num_layers=3 \
    --batch_size=32 \
    --max_seq_len=300 \
    --learning_rate=0.0003 \
    --learning_rate_decay_factor=0.95 \
    --optimizer="adam" \
    --flag_bidirect=True \
    --flag_generate=False \
    --flag_varlen=True \
    --random="random" \
    --prob_high=0.4 \
    --prob_in=0.9 \
    --prior=3 \
    --num_wit=5 \
    --num_top=5 \
    --flag_sum=0 \
    --model="lstm" \
    --keep_prob=0.9 