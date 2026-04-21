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

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
parser.add_argument('--model_path', type=str, default='/data2/Mamba_convolution/heatmap/slide/s_1_checkpoint.pth') 
parser.add_argument('--cluster_result_path', type=str, default='/mnt/data3/heatmap/tsne/clustering_results.txt')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = args.model_path
cluster_result_path = args.cluster_result_path

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

def load_cluster_scores(file_path):
    cluster_df = pd.read_csv(file_path, sep=None, engine='python')
    if cluster_df.shape[1] == 0:
        raise ValueError('cluster result file is empty')
    cluster_scores = cluster_df.iloc[:, 0].to_numpy().reshape(-1, 1)
    return cluster_scores

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
		#model.eval()
        with torch.no_grad():
            _, _, Y_hat, A, Y_prob = model(features)
            Y_hat = Y_hat.item()
            #A = A[Y_hat]
            A = A.view(-1, 1).cpu().numpy()
            
            # -------------------------------------------------------------
            # LOAD CLUSTER SCORES
            # -------------------------------------------------------------
            cluster_labels_all = load_cluster_scores(cluster_result_path)
            
            # Ensure the number of cluster labels matches the number of patches in this slide
            num_patches = A.shape[0]
            
            # Assuming the cluster_results.txt contains labels for ALL slides concatenated
            # OR just for this slide. If it's just for this slide, it should match `num_patches`.
            # If it's global, we need a way to index into it.
            # For now, if the lengths don't match, we take the first `num_patches` 
            # (assuming this script is run per slide and the file matches the slide).
            # We MUST ensure A only contains valid cluster labels (0-7).
            if len(cluster_labels_all) >= num_patches:
                A = cluster_labels_all[:num_patches]
            else:
                print(f"WARNING: Not enough cluster labels ({len(cluster_labels_all)}) for patches ({num_patches}). Padding with -1.")
                A = np.pad(cluster_labels_all, ((0, num_patches - len(cluster_labels_all)), (0, 0)), constant_values=-1)
                
            print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
            probs, ids = torch.topk(Y_prob, exp_args.n_classes)
            Y_probs = probs[-1].cpu().numpy()
            Y_hats = ids[-1].cpu().numpy()
            Y_hats_str = np.array([reverse_label_dict[idx] for idx in Y_hats])
        del features
        
        #if not os.path.isfile(block_map_save_path): 
        file = h5py.File(h5_path, "r")
        coords = file['coords'][:]
        file.close()
        asset_dict = {'attention_scores': A, 'coords': coords}
        block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        #os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('{}.csv'.format(exp_args.save_exp_code), index=False)
        
        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        # Generate single subtype maps correctly mapping scores
        subtype_save_dir = os.path.join(r_slide_save_dir, 'single_subtype_maps')
        os.makedirs(subtype_save_dir, exist_ok=True)
        # We enforce exactly 8 clusters (0 to 7) based on TSNE setting
        subtype_labels = np.arange(8)
        
        import tifffile

        for subtype_label in subtype_labels:
            # Generate a strict binary mask: 1.0 for the current subtype, 0.0 for all others
            # ONLY consider patches where the score exactly matches the valid subtype label (0-7)
            # This completely ignores -1 or any other invalid background padding values
            subtype_scores = (scores.reshape(-1) == subtype_label).astype(np.float32).reshape(-1, 1)
            
            # Save as TIFF to support pyramidal format for QuPath
            subtype_map_name = '{}_subtype_{}_mask.tiff'.format(slide_id, int(subtype_label))
            subtype_map_path = os.path.join(subtype_save_dir, subtype_map_name)
            if os.path.isfile(subtype_map_path):
                continue
            
            # Use alpha=1.0 and blank_canvas=False to retain H&E background where mask is active
            # Use custom colormap 'gray' but scale it so active regions (100.0) don't get completely overridden by cmap
            # Actually, to show original H&E patches for the subtype, we just need alpha=0.0 so cmap doesn't hide it, 
            # and blank_canvas=True would hide everything else. But CLAM's block_blending blends everywhere.
            # To purely show H&E for target patches and white for others:
            subtype_scores = subtype_scores * 100.0
            
            subtype_heatmap = drawHeatmap(subtype_scores, coords, slide_path, wsi_object=wsi_object,
                            cmap='gray', alpha=0.0, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=False)
            
            # Post-process: since alpha=0.0 shows full H&E everywhere in drawHeatmap, we must mask it manually.
            # A simpler CLAM-native way: use a colormap that is fully transparent for active, white for inactive.
            # But since we can't easily inject custom cmap objects here without rewriting, we use blank_canvas=True, alpha=0.0.
            # Wait, if alpha=0.0, the canvas (blank or not) shows through.
            # Let's use alpha=1.0 and a custom behavior by hacking binarize. 
            # If binarize=True, it only copies the color block over if score >= threshold. 
            # But the inactive areas will retain the base image (which is H&E if blank_canvas=False).
            # To get white background for inactive, we need blank_canvas=True.
            # Then active areas will get the color block. If cmap is 'gray' and score is 100, color block is white. 
            # We want active areas to be H&E. 
            # We must use block_blending manually or use a trick:
            # Let's generate a raw H&E image and a blank canvas, and combine them using the mask.
            
            # Since drawHeatmap might be tricky to force into "H&E for active, White for inactive", 
            # we can use alpha=0.0 (fully show canvas) and blank_canvas=True. But canvas is white everywhere.
            # Let's revert to a simpler way: drawHeatmap with binarize=True, blank_canvas=True, cmap='gray'.
            # Wait, you requested "保留相应亚群的patch的HE染色图片". 
            
            subtype_heatmap = drawHeatmap(subtype_scores, coords, slide_path, wsi_object=wsi_object,
                            cmap='jet', alpha=0.0, use_holes=True, binarize=True, vis_level=-1, blank_canvas=True,
                            thresh=0.5, patch_size=vis_patch_size, convert_to_percentiles=False)
            
            # To get H&E for active regions, we should set blank_canvas=False, and for inactive regions we want white.
            # Let's read the full H&E image and paste it onto a white canvas using the coordinates!
            
            # Use the patch level (which is much higher resolution) instead of vis_level (which is ~32x downsampled)
            # This ensures the output images have sufficient detail to see the H&E stain properly
            he_vis_level = patch_args.patch_level
            region_size = wsi_object.level_dim[he_vis_level]
            downsample = wsi_object.level_downsamples[he_vis_level]
            scale = [1/downsample[0], 1/downsample[1]] if isinstance(downsample, (tuple, list)) else [1/downsample, 1/downsample]
            
            # The coordinates in `coords` are typically level 0 coordinates.
            # patch_args.patch_size is the size at patch_args.patch_level.
            scaled_coords = np.ceil(coords * np.array(scale)).astype(int)
            
            # Check if there are any patches for this subtype to avoid unnecessary processing
            has_patches = np.any(subtype_scores > 0)
            
            if has_patches:
                # To prevent MemoryError (Killed) when creating a massive canvas in RAM, 
                # we use numpy memmap to create the canvas directly on disk.
                import tempfile
                import uuid
                temp_mmap_file = os.path.join(r_slide_save_dir, f"canvas_{uuid.uuid4().hex}.dat")
                
                # Shape is (height, width, 3) for RGB
                mmap_shape = (region_size[1], region_size[0], 3)
                try:
                    white_canvas = np.memmap(temp_mmap_file, dtype='uint8', mode='w+', shape=mmap_shape)
                    white_canvas[:] = 255 # Fill with white
                    
                    for idx in range(len(coords)):
                        if subtype_scores[idx] > 0:
                            # coords[idx] is the top_left coordinate at level 0
                            # We read the patch directly from the WSI using OpenSlide
                            pt = tuple(coords[idx])
                            patch_size_tuple = (patch_args.patch_size, patch_args.patch_size)
                            patch = wsi_object.wsi.read_region(pt, he_vis_level, patch_size_tuple).convert("RGB")
                            patch_arr = np.array(patch)
                            
                            # The paste coordinates must be scaled to match our canvas's level (he_vis_level)
                            paste_coord = scaled_coords[idx]
                            y_start, y_end = paste_coord[1], paste_coord[1] + patch_arr.shape[0]
                            x_start, x_end = paste_coord[0], paste_coord[0] + patch_arr.shape[1]
                            
                            # Ensure we don't go out of bounds
                            y_end = min(y_end, mmap_shape[0])
                            x_end = min(x_end, mmap_shape[1])
                            
                            patch_h = y_end - y_start
                            patch_w = x_end - x_start
                            
                            white_canvas[y_start:y_end, x_start:x_end] = patch_arr[:patch_h, :patch_w]
                    
                    # Flush changes to disk
                    white_canvas.flush()
                    
                    # Save the image as a pyramidal TIFF using tifffile
                    # tifffile can read directly from the memmap array, bypassing RAM limitations
                    with tifffile.TiffWriter(subtype_map_path, bigtiff=True) as tif:
                        options = dict(
                            photometric='rgb',
                            tile=(256, 256),
                            compression='jpeg',
                            resolution=(1.0, 1.0)
                        )
                        # Write the full resolution image
                        tif.write(white_canvas, subifds=4, **options)
                        
                        # Generate and write 4 pyramid levels
                        # We use cv2 for fast resizing if available, otherwise PIL
                        current_image = white_canvas
                        for level in range(4):
                            new_shape = (current_image.shape[0] // 2, current_image.shape[1] // 2)
                            if new_shape[0] < 256 or new_shape[1] < 256:
                                break
                            
                            try:
                                import cv2
                                downsampled = cv2.resize(current_image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
                            except ImportError:
                                from PIL import Image
                                downsampled = np.array(Image.fromarray(current_image).resize((new_shape[1], new_shape[0]), Image.Resampling.BILINEAR))
                                
                            tif.write(downsampled, **options)
                            current_image = downsampled
                finally:
                    # Clean up memmap and temporary file
                    del white_canvas
                    if os.path.exists(temp_mmap_file):
                        os.remove(temp_mmap_file)
            else:
                # If no patches, just create a tiny dummy white tiff to save space
                dummy_canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)
                with tifffile.TiffWriter(subtype_map_path, bigtiff=True) as tif:
                    tif.write(
                        dummy_canvas,
                        photometric='rgb',
                        tile=(256, 256),
                        compression='jpeg',
                        resolution=(1.0, 1.0)
                    )
                del dummy_canvas

        samples = sample_args.samples
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

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

        # Global labels mapping logic for main blockmap:
        # Instead of scaling by max_label in THIS slice (which might only be 1),
        # we should map the label directly to a global scale. We know you have max 8 clusters (0-7).
        GLOBAL_MAX_CLUSTERS = 7 # Assuming labels 0 to 7 based on earlier TSNE logic optimal_clusters=8
        
        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            # Map labels to [0, 100] globally
            scaled_scores = ((scores / GLOBAL_MAX_CLUSTERS) * 99.0) + 1.0
                
            heatmap = drawHeatmap(scaled_scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
        
            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, 
                                img_transforms=img_transforms,
                                clam_pred=Y_hats[0], model=model, 
                                feature_extractor=feature_extractor, 
                                batch_size=exp_args.batch_size, **wsi_kwargs, 
                                attn_save_path=save_path,  ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        with h5py.File(save_path, 'r') as file:
            file = h5py.File(save_path, 'r')
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                        int(heatmap_args.blur), 
                                                                                        int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                        float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                        int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        
        else:                                                                                                                                                   
            # Also apply scaling for the final heatmap if convert_to_percentiles is False
            final_scores = scores
            if not heatmap_vis_args['convert_to_percentiles']:
                GLOBAL_MAX_CLUSTERS = 7 # Use the same global max to keep color maps consistent across slides
                final_scores = ((scores / GLOBAL_MAX_CLUSTERS) * 99.0) + 1.0
                    
            heatmap = drawHeatmap(final_scores, coords, slide_path, wsi_object=wsi_object,  
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                                  binarize=heatmap_args.binarize, 
                                    blank_canvas=heatmap_args.blank_canvas,
                                    thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=patch_args.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            if heatmap_args.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
        
        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
