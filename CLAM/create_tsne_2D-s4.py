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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

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
    
    data_list = []
    label_list = []
    #optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    #optimizer.zero_grad()
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
        features.requires_grad = False
        
        model.eval()
        #raise ValueError("已通过")
        #features = features.cuda(non_blocking=True)
        #features.requires_grad = True
        #optimizer.zero_grad()
        #_, _, _, outputs, _ = model(features)
        #features = features.requires_grad_()
        with torch.no_grad():
            _, _, _, _, h = model(features)
            h = h.cpu().numpy()
        #data = outputs.squeeze(0)
        #random_indices = torch.randperm(data.size(0))
        #print(data.shape)
        #print(A.shape)
        #raise ValueError("已通过")
        #A = torch.tensor(A, dtype=torch.float32)
        #A = torch.abs(A)
        #_, top_indices = torch.topk(A.squeeze(), 10)
        #data = data[top_indices]
        #print(data.shape)
        #print(A)
        #raise ValueError("已通过")
        #data = data[random_indices[:10]]
        #print(data.shape)
        #raise ValueError("已通过")
        #datashape = data.shape[0]
        print("lable::")
        print(label)
        data_list.append(h)
        #label_list.append([label] * datashape)
        del features, h
        #del features, feature, A_, A, B_, B, outputs
        torch.cuda.empty_cache() 
    data_array = np.concatenate(data_list, axis=0)
    #data_array = np.array(data_list.detach().cpu().numpy())
    # Reduce dimensionality to 2 components
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(data_array)
    
    # Determine the optimal number of clusters
    def calculate_optimal_clusters(data, max_clusters=10):
        best_num_clusters = 0
        best_silhouette_score = -1
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = n_clusters
        return best_num_clusters
    
    optimal_clusters = calculate_optimal_clusters(tsne_results)
    optimal_clusters = 8
    
    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    clusters = kmeans.fit_predict(tsne_results)
    # Visualization
    def plot_clusters(data, cluster_labels, palette, output_dir, output_filename):
        plt.figure(figsize=(10, 8))
        for i in range(len(palette)):
            plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], 
                        label=f'Cluster {i}', marker='o', s=30, color=palette[i])
        plt.title('t-SNE Clustering')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
    
        # Save the plot to the specified location
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close()
    
    # Define the output directory and filename for the image and clustering results
    output_dir = '/mnt/data3/heatmap/tsne/'  # Replace with your path
    output_filename = 'tsne_clustering.pdf'  # Name of the image file
    results_filename = 'clustering_results.txt'  # Name of the text file for clustering results
    
    # Palette of colors for the clusters
    palette = ['#93CABE', '#5AEEEF', '#4AC9CA', '#D6208F', '#8AABCA', '#952E33', '#BEE060', '#D389B8', '#7934D6', '#C42326', '#4B80B4', '#74BA67', '#4A94F2', '#FAFF68', '#B8A17A']
    
    # Call the plotting function to save the plot
    plot_clusters(tsne_results, clusters, palette, output_dir, output_filename)
    
    # Save clustering results to a text file
    results_path = os.path.join(output_dir, results_filename)  
    with open(results_path, 'w') as file:  
        file.write('Sample_Name\tComponent_1\tComponent_2\tCluster_Label\n')  # Header line  
        for i, label in enumerate(clusters):  
            file.write(f"{process_stack.loc[i, 'slide_id']}\t{tsne_results[i, 0]}\t{tsne_results[i, 1]}\t{label}\n")