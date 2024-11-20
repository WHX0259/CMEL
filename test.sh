#!/bin/bash
export MASTER_PORT=6256
export CUDA_VISIBLE_DEVICES=0

model_type="ClinicalImageBaseClusterDistancePlusGatingModel"
logit_method='ClinicalImageBaseClusterDistancePlusGatingModel'
criterion_type="mixed"
for fold in {0..9}
    do
    next_fold=$(($fold + 1))
    #/data/wuhuixuan/code/Multi_Modal_MoE/out/ClinicalImageALBEFClusterGatingModel/num_expert3/mixed/1/1_1_1/var/No_Use_Weights/Focal_weight_1.0/Focal_weight_1.0/Interaction_0.2/lfv_0.0/sim_0.02/dis_0.02/train/ClinicalImageALBEFClusterGatingModel/model_best.pth.tar
    resume="/data/wuhuixuan/code/Multi_Modal_MoE/out/$model_type/num_expert3/$criterion_type/${next_fold}/1_1_1/var/No_Use_Weights/Focal_weight_1.0/Focal_weight_1.0/Interaction_0.2/lfv_0.0/sim_0.02/dis_0.02/train/$model_type/model_best.pth.tar"
    python main.py --fold $fold --model_type "$model_type" --batch-size 8 --epoch 100 --logit_method "$logit_method" --criterion_type "$criterion_type" --extra_dim 22 --diversity_metric "var" --sim_weight 0.02 --dis_weight 0.02 --evaluate --display_experts True --resume $resume 
    done
