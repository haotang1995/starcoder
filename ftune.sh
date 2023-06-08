#!/bin/bash

python -m torch.distributed.launch \
    --nproc_per_node 1 finetune/finetune.py \
    --model_path="bigcode/starcoder"\
    --dataset_name="ArmelR/stack-exchange-instruction"\
    --subset="data/finetune"\
    --split="train"\
    --size_valid_set 10000\
    --streaming \
    --seq_length 2048\
    --max_steps 1000\
    --batch_size 1\
    --input_column_name="question"\
    --output_column_name="response"\
    --gradient_accumulation_steps 16\
    --learning_rate 1e-4\
    --lr_scheduler_type="cosine"\
    --num_warmup_steps 100\
    --weight_decay 0.05\
    --output_dir="./checkpoints" 
