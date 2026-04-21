from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from torch import optim as optim
from timm.utils import AverageMeter

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
parser.add_argument('--model_path', type=str, default='/data2/Mamba_convolution/heatmap/slide/s_1_checkpoint.pth') 
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = args.model_path

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

if __name__ == '__main__':
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print (value_key + " : " + str(value_value))
        else:
            print ('\n'+key + " : " + str(value))
            
    #decision = input('Continue? Y/N ')
    #if decision in ['Y', 'y', 'Yes', 'yes']:
    #    pass
    #elif decision in ['N', 'n', 'No', 'NO']:
    #    exit()
    #else:
    #    raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = args['encoder_arguments']
    encoder_args = argparse.Namespace(**encoder_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]


    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    model = torch.load(model_path.format('int'), map_location=torch.device('cuda'))

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    print('Done!')

    label_dict =  data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
    

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
    
    meter = AverageMeter()
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    optimizer.zero_grad()
    for i in tqdm(range(len(process_stack))):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name+=data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
    

        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path) :
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                            model=model, 
                                            feature_extractor=feature_extractor, 
                                            img_transforms=img_transforms,
                                            batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
                                            attn_save_path=None, feat_save_path=h5_path, 
                                            ref_scores=None)				
        
        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features 
        features = torch.load(features_path)
        process_stack.loc[i, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)
        features = features.to(device)
        
        model.eval()
        #raise ValueError("已通过")
        features = features.cuda(non_blocking=True)
        features.requires_grad = True
        optimizer.zero_grad()
        feature = model.feature(features)
        feature = feature.squeeze()
        #outputs = model.DFF(A.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1),B.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1, 3)#给特征加权重
        outputs = feature.unsqueeze(-1).unsqueeze(0)
        #print(outputs.shape)
        #outputs = outputs.reshape(1, -1, 16, 32)
        #print(outputs.shape)
        #raise ValueError("已通过")
        #outputs = outputs.squeeze(0).unsqueeze(2).unsqueeze(3)
        out_size = outputs.size()
        #print(out_size)
        #raise ValueError("已通过")
        central_point = torch.nn.functional.leaky_relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        #print(central_point)
        grad = torch.autograd.grad(central_point, features)
        #print(grad[0].shape)
        #raise ValueError("已通过")
        grad = grad[0]
        grad = grad.unsqueeze(2).unsqueeze(3)
        grad = grad.reshape(1, -1, 32, 32)
        #print(grad.shape)
        #raise ValueError("已通过")
        grad = torch.nn.functional.relu(grad)
        #print(grad.shape)
        aggregated = grad.sum((0, 1))
        #print(aggregated.shape)
        contribution_scores = aggregated.cpu().numpy()
        #print(contribution_scores.shape)
        #raise ValueError("已通过")
        meter.update(contribution_scores)
        #print(meter.count)
    
        if meter.count == len(process_stack):
            np.save('/mnt/data3/heatmap/erf/temp', meter.avg)
            print(meter.avg.shape)
            print('Saved')