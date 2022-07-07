#!/usr/bin bash


for s in 42
do
    python Main.py \
        --dataset banking \
        --known_cls_ratio 0.9 \
        --cluster_num_factor 2 \
        --seed $s \
        --data_augumentation_type 2 \
        --freeze_bert_parameters \
        --save_model \
        --method KT \
        --train_batch_size 400 \
        --pre_train_batch_size 128 \
        --gpu_id 1 \
        --pretrain \
        --pretrain_dir pretrain_models_v2_0.9_cescl_seed42_banking

done
