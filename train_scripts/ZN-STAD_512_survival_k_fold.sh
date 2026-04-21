model_names='mamba_attn trans_mil mamba_mil'
#model_names='mamba_attn trans_mil mamba_mil'
# model_names='mamba_mil'
backbones="PLIP"
# backbones='resnet50'
# resnet50 CONCH

declare -A in_dim
#in_dim["resnet50"]=1024
in_dim["RN50-B"]=1024
in_dim["UNI"]=1024
#in_dim["CONCH"]=1024
in_dim["PLIP"]=768

declare -A gpus
#gpus["mean_mil"]=1
#gpus["max_mil"]=0
#gpus["att_mil"]=1
gpus["trans_mil"]=1
#gpus['s4model']=0
gpus['mamba_mil']=1
gpus['mamba_attn']=1

cancers='STAD'

lr='2e-4'
#lr='2e-4'
reg='1e-3'
drop_out='0.35'
mambamil_rate='10'
mambamil_layer='2'
mambamil_type='SRMamba'

for cancer in $cancers
    do
    task="TCGA_STAD_survival"
    #Change to your path
    data_root_dir0="/data2/Mamba_convolution/dataset/Zhongnan-STAD/tiles-20x-s256/feature"
    results_dir="/data2/Mamba_convolution/trainresult/Zhongnan_STAD_survival"
    for model in $model_names
    do
        for backbone in $backbones
        do
            exp=$model"/"$backbone
            data_root_dir=$data_root_dir0"/"$backbone"/"
            echo $data_root_dir
            echo $exp", GPU is:"${gpus[$model]}
            export CUDA_VISIBLE_DEVICES=${gpus[$model]}
            # k_start and k_end, only for resuming, default is -1
            export WANDB_API_KEY="${WANDB_API_KEY:-}"
            echo "3" | python main_survival.py \
                --drop_out $drop_out\
                --lr $lr \
                --reg $reg \
                --k 5 \
                --exp_code $exp \
                --max_epochs 35 \
                --task $task \
                --results_dir $results_dir \
                --model_type $model \
                --split_dir "/data2/Mamba_convolution/split/Zhongnan_STAD_survival_kfold" \
                --data_root_dir $data_root_dir \
                --in_dim ${in_dim[$backbone]} \
                --k_fold True \
                --mambamil_rate $mambamil_rate \
                --mambamil_layer $mambamil_layer \
                --mambamil_type $mambamil_type
        done
    done
done
