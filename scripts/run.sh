#!/usr/bin bash


for s in 35
do
    python Main.py \
        --dataset virtual_assistant \
        --dataset_test dstc \
        --bert_model all-mpnet-base-v2 \
        --seed $s \
        --data_augumentation_type 2 \
        --freeze_bert_parameters \
        --save_model \
        --method DKT \
        --train_batch_size 400 \
        --gpu_id 1 \
        --pretrain \
        --num_train_epochs 70 \
        --pretrain_dir pretrain_models

done
