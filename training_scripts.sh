#!/bin/bash

# WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,3,4 --masterport 1234
# torchrun --nproc_per_node=4 \
python finetune.py  \
--base_model 'decapoda-research/llama-7b-hf' \
--output_dir './lora-alpaca' \
--batch_size 6 \
--micro_batch_size 6 \
--cutoff_len 1024 \
--num_epochs 2 \
--learning_rate 2e-4 \
--val_set_size 2000 \
--wandb_run_name 'run_43' \
--group_by_length
