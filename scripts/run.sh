#!/usr/bin bash


for s in 29
do
    python Main.py \
        --dataset banking \
        --known_cls_ratio 0.8 \
        --cluster_num_factor 2 \
        --seed $s \
        --data_augumentation_type 2 \
        --freeze_bert_parameters \
        --save_model \
        --method DKT \
        --train_batch_size 400 \
        --pre_train_batch_size 128 \
        --gpu_id 0 \
        --pretrain \
        --pretrain_dir pretrain_models_v1_0.8_cescl_seed42

done
