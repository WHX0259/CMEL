# CMEL
Clustering Multimodal Ensemble Learning for Predicting Gastric Cancer Chemotherapy Efficacy

# Requirements
  python 3.9
  
  torch 2.0.1
  
  torchaudio 2.0.2 
  
  torchvision 0.15.2    
  
  cuda 11.8
    
Please refer to the specific details in requirements.txt.

# Running the code
1. Setting up the Python environment

    The code has been tested with Python 3.9.19. The necessary libraries can be installed using pip with the following command:
    `
    #from SMEL/
    pip install -r requirements.txt
    `
2. Dataset setting

    A dataset function is required to read your own data. The SMEL model needs both image data and radiomics data. The dimension of the extracted and filtered radiomics data determines the setting of the parameter `extra_dim`.

3. Running the code

    You can now run the code on your dataset. You need to write a shell script file (with the extension `.sh`). Below is an example. You can set your parameters in the shell script file.

   **Train**:
   ```
    #!/bin/bash
    export MASTER_PORT=6256
    export CUDA_VISIBLE_DEVICES=0
    model_type="ClinicalImageBaseClusterDistancePlusGatingModel"
    logit_method='ClinicalImageBaseClusterDistancePlusGatingModel'
    criterion_type="mixed"
    for fold in {0..9}
        do
        next_fold=$(($fold + 1))
        resume="/data/wuhuixuan/code/Multi_Modal_MoE/out/$model_type/num_expert3/$criterion_type/${next_fold}/1_1_1/var/No_Use_Weights/Focal_weight_1.0/Focal_weight_1.0/Interaction_0.2/lfv_0.0/sim_0.02/dis_0.02/train/$model_type/model_best.pth.tar"
        python main.py --fold $fold --model_type "$model_type" --batch-size 8 --epoch 100 --logit_method "$logit_method" --criterion_type "$criterion_type" --extra_dim 22 --diversity_metric "var" --sim_weight 0.02 --dis_weight 0.02 #--evaluate --display_experts True --resume $resume 
        done

   ```
   **Test**:
    ```
    #!/bin/bash
    export MASTER_PORT=6256
    export CUDA_VISIBLE_DEVICES=0
    model_type="ClinicalImageBaseClusterDistancePlusGatingModel"
    logit_method='ClinicalImageBaseClusterDistancePlusGatingModel'
    criterion_type="mixed"
    for fold in {0..9}
        do
        next_fold=$(($fold + 1))
        resume="/data/wuhuixuan/code/Multi_Modal_MoE/out/$model_type/num_expert3/$criterion_type/${next_fold}/1_1_1/var/No_Use_Weights/Focal_weight_1.0/Focal_weight_1.0/Interaction_0.2/lfv_0.0/sim_0.02/dis_0.02/train/$model_type/model_best.pth.tar"
        python main.py --fold $fold --model_type "$model_type" --batch-size 8 --epoch 100 --logit_method "$logit_method" --criterion_type "$criterion_type" --extra_dim 22 --diversity_metric "var" --sim_weight 0.02 --dis_weight 0.02 --evaluate --display_experts True --resume $resume 
        done
    
    ```
    
    Note: If you do not need ten-fold cross-validation, you can set the fold to any number or leave it unset. You should remove the `for` loop statement in the `.sh` file.
    Note: `--evaluate` sets the evaluation mode, `--resume` is the path to the model parameters file, and if you need to output images from each expert model, set `--display_experts` to True.

