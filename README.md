## Introduction

DSTC11 评测代码框架，目前可支持两类基线：

(1) K-means [_**baseline**_]：backbone网络提取表征，然后K-means聚类

(2) 对比聚类: 在backbone网络后面加上两个MLP分支，一个分支进行实例级对比学习，另一个分支进行聚类 

所有上述方法都支持pretrain和non-pretrain两个版本。pretrain意味着我们在做上述聚类优化之前，第一阶段先用一些开源意图识别数据集进行预训练。

## Usage

K-means聚类（没有预训练阶段）
```
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
        --method K-means \
        --train_batch_size 400 \
        --gpu_id 1 \
        --num_train_epochs 70 \
        --pretrain_dir pretrain_models

done
```

K-means聚类（+预训练阶段）
```
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
        --method K-means \
        --train_batch_size 400 \
        --gpu_id 1 \
        --pretrain
        --num_train_epochs 70 \
        --pretrain_dir pretrain_models

done
```

SCCL（+预训练阶段）
```
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
```


## Results


|      | ACC | ARI | NMI |
|------------|:-----------:|:-----------:|:-----------:|
| K-means w/o. pretrain        |     46.22    |     32.00    |     55.44    |
| K-means w. pretrain |     55.68    |      41.19      |     62.83    |
| SCCL w/o. pretrain        |     51.78    |      37.11      |     59.39    |
| SCCL w. pretrain       |     **66.22**    |     **49.28**    |     **68.59**    |
