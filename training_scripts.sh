#!/bin/bash

# WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,3,4 --masterport 1234
# torchrun --nproc_per_node=4 \
python finetune.py  \
--base_model 'decapoda-research/llama-7b-hf' \
--output_dir './lora-alpaca_5' \
--batch_size 17 \
--micro_batch_size 17 \
--cutoff_len 1024 \
--num_epochs 2 \
--learning_rate 3e-4 \
--val_set_size 2000 \
--wandb_run_name 'run_41' \
# --resume_from_checkpoint './lora-alpaca_3/checkpoint-10000' \
--group_by_length
