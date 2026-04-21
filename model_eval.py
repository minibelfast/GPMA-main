from models.MambaAttn import MambaAttn
from sksurv.metrics import concordance_index_censored
import pandas as pd
import numpy as np
import os
import torch
import argparse

from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.survival_utils import get_split_loader
from utils.survival_core_utils import summary_survival

# 创建 ArgumentParser 对象  
parser = argparse.ArgumentParser()  

# 添加命令行参数 
parser.add_argument('--kfold', type=int, default=5)  
parser.add_argument('--train_path', type=str, default='/data2/Mamba_convolution/trainresult/TCGA_STAD_survival/mamba_attn/')   
parser.add_argument('--slide_path0', type=str, default='/data2/Mamba_convolution/dataset/TCGA_STAD/tiles-20x-s256/feature/')  
parser.add_argument('--split_dir', type=str, default="/data2/Mamba_convolution/split/ALL_STAD_survival_kfold", 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--save_path', type=str, default='/data2/Mamba_convolution/model_eval/')  
parser.add_argument('--csv_path', type=str, default='STAD') 
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--backbone', type=str, default="CONCH")
parser.add_argument('--patch_size', type=str, default='')


args = parser.parse_args()  

kfold = args.kfold
train_path = args.train_path
csv_path = os.path.join('/data1/single_cell/AI/MambaMIL-main/dataset_csv', f"{args.csv_path}_processed.csv")
save_path = args.save_path
data_root_dir = args.slide_path0 + args.backbone + '/pt_files/'
mode = args.mode

results_list = []  
models = {} 
c_index = np.zeros(kfold)
patient_results_dict = {} 
for i in range(kfold):
    model_path = train_path + args.backbone + "_s1/s_" + str(i) + "_checkpoint.pth"
    model_name = 'model' + '_' + str(i)  
    models[model_name] = torch.load(model_path.format('int'), map_location=torch.device('cuda'))
    
    dataset = Generic_MIL_Survival_Dataset(csv_path = csv_path,
                                           mode = mode,
                                           apply_sig = args.apply_sig,
                                           data_dir= data_root_dir, #! cluster.pkl should be as same as data_dir
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
    
    train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size,
                                                                     from_id=False,csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

    
    datasets = (train_dataset, val_dataset, test_dataset)
    train_split, val_split, test_split = datasets
    test_loader = get_split_loader(test_split, testing = False, mode=mode, batch_size=1)

    patient_results, c_index[i] = summary_survival(models[model_name], test_loader, 4)
    if i == 0:
        patient_results = pd.DataFrame.from_dict(patient_results, orient='index')  
        patient_results_dict[f'patient_results{i}'] = patient_results.rename(columns={'risk': f'risk{i}'})
    else:
        patient_results = {key: {'slide_id': value['slide_id'], 'risk': value['risk']} for key, value in patient_results.items()} 
        patient_results = pd.DataFrame.from_dict(patient_results, orient='index')  
        patient_results_dict[f'patient_results{i}'] = patient_results.rename(columns={'risk': f'risk{i}'})
        
    
merged_df = pd.DataFrame()
# 合并所有 patient_results{i} 表格  
for i in range(kfold):  
    # 读取 patient_results{i} 表格，假设数据存储在名为 patient_results{i}.csv 的文件中  
    df_i = patient_results_dict[f'patient_results{i}']
    
    # 如果是第一个表格，直接赋值给 merged_df  
    if merged_df.empty:  
        merged_df = df_i  
    else:  
        merged_df = pd.concat([merged_df, df_i], axis=1)
# 保存到 CSV 文件  
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]   
merged_df.to_csv(save_path+'output.csv', index=False)  
print("OUTPUT已保存：output.csv")

df_c_index = pd.DataFrame({'c_index': c_index}) 
df_c_index.to_csv(save_path+'c_index.csv', index=False)  
print("c_index已保存：c_index.csv")  