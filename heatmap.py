from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from CLAM.utils.utils import *
from math import floor
#from utils.eval_utils import initiate_model as initiate_model
#from models.model_clam import CLAM_MB, CLAM_SB
#from models import get_encoder
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
#from wsi_core.batch_process_utils import initialize_df
#from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
#from wsi_core.wsi_utils import sample_rois
#from utils.file_utils import save_hdf5
from tqdm import tqdm
import ast 
import pickle
import h5py

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path


parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default="MambaAttn",
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--train_path', type=str, default='/data2/Mamba_convolution/trainresult/TCGA_STAD_survival/mamba_attn/')   
parser.add_argument('--backbone', type=str, default="UNI")
parser.add_argument('--modelnumber', type=str, default=0)#kfold数目中选择的最佳模型编号比如5fold则是0-4
parser.add_argument('--label_dict', type=str, default="{'High': 4, 'Middlehigh': 3, 'Middlelow': 2, 'Low': 1}")  
parser.add_argument('--slide_ext', type=str, default=".svs")
parser.add_argument('--process_list', type=str, default="/data2/Mamba_convolution/heatmap/heatmap_demo_dataset.csv")
parser.add_argument('--data_dir', type=str, default="/data2/Mamba_convolution/heatmap/slide")
parser.add_argument('--n_classes', type=str, default=4)
parser.add_argument('--samples', type=str,   
                    default=json.dumps([{"name": "topk_high_attention", "sample": True, "seed": 1, "k": 15, "mode": "topk"}]))
args = parser.parse_args()



if __name__ == '__main__':
    model_path = args.train_path + args.backbone + "_s1/s_" + str(args.modelnumber) + "_checkpoint.pth"
    model = torch.load(model_path.format('int'), map_location=torch.device('cuda'))
    label_dict = ast.literal_eval(args.label_dict) 
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
    process_stack = pd.read_csv(args.process_list)

    #循环
    i = 1
    slide_name = process_stack.loc[i, 'slide_id']
    if args.slide_ext not in slide_name:
        slide_name+=args.slide_ext
    print('\nprocessing: ', slide_name)	
    try:
        label = process_stack.loc[i, 'label']
    except KeyError:
        label = 'Unspecified'
    slide_id = slide_name.replace(args.slide_ext, '')
    if not isinstance(label, str):
        grouping = reverse_label_dict[label]
    else:
        grouping = label 
    top_left = None
    bot_right = None
    print('slide id: ', slide_id)
    print('top left: ', top_left, ' bot right: ', bot_right)
    if isinstance(args.data_dir, str):
        slide_path = os.path.join(args.data_dir, slide_name)
    elif isinstance(args.data_dir, dict):
        data_dir_key = process_stack.loc[i, args.data_dir_key]
        slide_path = os.path.join(args.data_dir[data_dir_key], slide_name)
    else:
        raise NotImplementedError
    #print('Initializing WSI object')
    #mask_file = os.path.join('/data2/Mamba_convolution/heatmap/', slide_id+'_mask.pkl')
    #seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
    #                  'keep_ids': 'none', 'exclude_ids':'none'}
    #filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    #wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
    #print('Done!')

    # load features 
    features = torch.load('/data2/Mamba_convolution/heatmap/slide/TCGA-3M-AB46-01Z-00-DX1.70F638A0-BDCB-4BDE-BBFE-6D78A1A08C5B.pt')
    process_stack.loc[i, 'bag_size'] = len(features)
    features = features.to(device)
    model.eval()
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, _ = model(features)
        Y_hat = Y_hat.item()
        A = A[Y_hat]
        A = A.view(-1, 1).cpu().numpy()
        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        probs, ids = torch.topk(Y_prob, k)
        Y_probs = probs[-1].cpu().numpy()
        Y_hats = ids[-1].cpu().numpy()
        Y_hats_str = np.array([reverse_label_dict[idx] for idx in ids])
    del features
    file = h5py.File('/data2/Mamba_convolution/heatmap/slide/TCGA-3M-AB46-01Z-00-DX1.70F638A0-BDCB-4BDE-BBFE-6D78A1A08C5B.h5', "r")
    coords = file['coords'][:]
    file.close()
    asset_dict = {'attention_scores': A, 'coords': coords}
    print(asset_dict)
    
    block_map_save_path = os.path.join('/data2/Mamba_convolution/heatmap/', '{}_blockmap.h5'.format(slide_id))
    block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
    
    for c in range(args.n_classes):
        process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
        process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

    if data_args.process_list is not None:
        process_stack.to_csv('/data2/Mamba_convolution/heatmap/result/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
    else:
        process_stack.to_csv('/data2/Mamba_convolution/heatmap/result/{}.csv'.format(exp_args.save_exp_code), index=False)

    file = h5py.File(block_map_save_path, 'r')
    dset = file['attention_scores']
    coord_dset = file['coords']
    scores = dset[:]
    coords = coord_dset[:]
    file.close()

    samples = args.samples
    for sample in samples:
        if sample['sample']:
            tag = "label_{}_pred_{}".format(label, Y_hats[0])
            sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
            os.makedirs(sample_save_dir, exist_ok=True)
            print('sampling {}'.format(sample['name']))
            sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
            for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


