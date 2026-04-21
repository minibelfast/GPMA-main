import pandas as pd  
import numpy as np  
import os  
import argparse  

def create_splits(csv_path, k, save_dir):  
    # 读取 CSV 文件  
    df = pd.read_csv(csv_path)  

    # 按照 status 列分组  
    group_0 = df[df['status'] == 0].copy()  
    group_1 = df[df['status'] == 1].copy()  

    # 按照 time 列从高到低排序  
    group_0.sort_values(by='time', ascending=False, inplace=True)  
    group_1.sort_values(by='time', ascending=False, inplace=True)  
    
    # 为每组样本进行标记  
    for group, group_name in zip([group_0, group_1], ['group_0', 'group_1']):  
        n_samples = len(group)  
        # 计算组内的标记  
        folds = np.tile(np.arange(1, k + 1), n_samples // k + 1)[:n_samples]
        # 随机打乱每个折叠的标记  
        for i in range(n_samples // k):  
            start_index = i * k  
            end_index = start_index + k  
            np.random.shuffle(folds[start_index:end_index])  
        # 对于不足 k 的样本，标记为 0  
        if n_samples % k != 0:  
            folds[-(n_samples % k):] = 0  # 将最后不足 k 的样本标记为 0  
        
        # 将折叠标记赋值给 group  
        group['fold'] = folds 
    
    # 处理第三组  
    if len(group_0) > len(group_1):  
        group_0_zero = group_0[group_0['fold'] == 0]  
        group_1_zero = group_1[group_1['fold'] == 0]  
        group_3 = pd.concat([group_0_zero, group_1_zero], ignore_index=True)  
        
        # 取出 status 为 0 的样本  
        group_3_status_0 = group_3[group_3['status'] == 0]  
        
        # 获取有效样本数量  
        n_samples_status_0 = len(group_3_status_0)  
        
        # 标记为 1 到 n，同时打乱顺序  
        if n_samples_status_0 > 0:  
            status_0_labels = np.arange(1, n_samples_status_0 + 1)  
            np.random.shuffle(status_0_labels)  
            # 使用 .loc 来更新  
            group_3.loc[group_3['status'] == 0, 'fold'] = status_0_labels  
        
        # 处理 status 为 1 的样本  
        group_3_status_1 = group_3[group_3['status'] == 1]  
        n_status_1_samples = len(group_3_status_1)  
        
        # 获取当前最大的 new_fold 值，以便从下一个值开始编码 status 为 1 的样本  
        max_new_fold = group_3['fold'].max() if 'fold' in group_3 else 0  
        
        # 标记 status 为 1 的样本  
        # 标记 status 为 1 的样本  
        if n_status_1_samples > 0:  
            # 计算下一个编码开始的数字  
            k_start = max_new_fold + 1   
        
            # 生成新的编码  
            if k_start + n_status_1_samples <= k:  
                # 如果 k_start + n_status_1_samples 小于等于 k  
                k_count = np.arange(k_start, k_start + n_status_1_samples)  
            else:  
                # 否则从 k_start 到 k, 然后从 1 到 n_status_1_samples - k + k_start  
                k_count = np.concatenate((np.arange(k_start, k + 1), np.arange(1, n_status_1_samples - (k - k_start) + 1)))  
        
            np.random.shuffle(k_count)  # 打乱顺序  
        
            # 更新 group_3 的 new_fold  
            group_3.loc[group_3['status'] == 1, 'fold'] = k_count[:n_status_1_samples]  
    else:  
        group_0_zero = group_0[group_0['fold'] == 0]  
        group_1_zero = group_1[group_1['fold'] == 0]  
        group_3 = pd.concat([group_0_zero, group_1_zero], ignore_index=True)  
        group_3 = group_3.drop(columns=['fold'])
        
        # 取出 status 为 0 的样本  
        group_3_status_0 = group_3[group_3['status'] == 1]  
        
        # 获取有效样本数量  
        n_samples_status_0 = len(group_3_status_0)  
        
        # 标记为 1 到 n，同时打乱顺序  
        if n_samples_status_0 > 0:  
            status_0_labels = np.arange(1, n_samples_status_0 + 1)  
            np.random.shuffle(status_0_labels)  
            # 使用 .loc 来更新  
            group_3.loc[group_3['status'] == 1, 'fold'] = status_0_labels  
        
        # 处理 status 为 1 的样本  
        group_3_status_1 = group_3[group_3['status'] == 0]  
        n_status_1_samples = len(group_3_status_1)  
        
        # 获取当前最大的 new_fold 值，以便从下一个值开始编码 status 为 1 的样本  
        max_new_fold = group_3['fold'].max() if 'fold' in group_3 else 0  
        
        # 标记 status 为 1 的样本  
        # 标记 status 为 1 的样本  
        if n_status_1_samples > 0:  
            # 计算下一个编码开始的数字  
            k_start = max_new_fold + 1   
        
            # 生成新的编码  
            if k_start + n_status_1_samples <= k:  
                # 如果 k_start + n_status_1_samples 小于等于 k  
                k_count = np.arange(k_start, k_start + n_status_1_samples)  
            else:  
                # 否则从 k_start 到 k, 然后从 1 到 n_status_1_samples - k + k_start  
                k_count = np.concatenate((np.arange(k_start, k + 1), np.arange(1, n_status_1_samples - (k - k_start) + 1)))  
        
            np.random.shuffle(k_count)  # 打乱顺序  
        
            # 更新 group_3 的 new_fold  
            group_3.loc[group_3['status'] == 0, 'fold'] = k_count[:n_status_1_samples] 

    # 合并所有组  
    # 去除 group_0 和 group_1 中 fold 等于 0 的样本  
    group_0_filtered = group_0[group_0['fold'] != 0]  
    group_1_filtered = group_1[group_1['fold'] != 0]  

    # 合并过滤后的 group_0, group_1 和 group_3  
    final_df = pd.concat([group_0_filtered, group_1_filtered, group_3], ignore_index=True) 

    # 创建保存目录  
    os.makedirs(save_dir, exist_ok=True)  

    for fold in range(1, k + 1):  
        # 创建训练集和验证集  
        train_df = final_df[final_df['fold'] != fold]  
        val_df = final_df[final_df['fold'] == fold]  
        # 创建一个新的 DataFrame，包含三列  
        combined_df = pd.DataFrame({  
            'train': train_df['slide_path'].reset_index(drop=True),  
            'val': val_df['slide_path'].reset_index(drop=True),  
            'test': val_df['slide_path'].reset_index(drop=True)  # 假设 test 列与 val 列相同  
        })  
        
        # 保存为 CSV 文件  
        combined_df.to_csv(os.path.join(save_dir, f'splits_{fold - 1}.csv'), index=True)  
        

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')  
    parser.add_argument('--k', type=int, required=True, help='Number of folds for cross-validation')  
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the splits')  
    
    args = parser.parse_args()  
    
    create_splits(args.csv_path, args.k, args.save_dir)