
from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import os
import glob
from tqdm import tqdm
import joblib
import yaml

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion, SkeletonMotion_dof
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# def get_weight_from_class(labels, label_abstract):
#     bodypart = label_abstract['Body part']
#     typeofmove = label_abstract['Type of movement']
#     simpledynamic = label_abstract['Simple dynamic actions']
#     static = label_abstract['Static actions']
#     objectinteration = label_abstract['Object interaction']
#     bodyinteraction = label_abstract['Body part interaction']
#     activities = label_abstract['Activity']
#     abstract = label_abstract['Abstract actions']

#     weight = 0.1
#     for label in labels:
#         if label in simpledynamic:
#             weight = 1.0
#     for label in labels:
#         if label in static:
#             weight = 0.0
#     return weight

def get_weight_from_class(labels, label_abstract):
    bodypart = label_abstract['Body part']
    typeofmove = label_abstract['Type of movement']
    simpledynamic = label_abstract['Simple dynamic actions']
    static = label_abstract['Static actions']
    objectinteration = label_abstract['Object interaction']
    bodyinteraction = label_abstract['Body part interaction']
    activities = label_abstract['Activity']
    abstract = label_abstract['Abstract actions']

    weight = 0.0
    selected_act = ['jump', 'stand up']#['run'] #['walk', 'run']

    for label in labels:
        if label in selected_act:
            weight = 1.0
    return weight


def main_getweight():
    # load motion raw
    skeleton_info = joblib.load('joints_info.pkl')
    skeleton = SkeletonTree(
                            node_names=skeleton_info['joints_name'],
                            parent_indices=torch.from_numpy(np.array(skeleton_info['parents'], dtype=np.int32)),
                            local_translation=torch.from_numpy(np.array(skeleton_info['loc_joints_pos'], dtype=np.float32)),
                            )
    root_path = '/home/datassd/yuxuan'
    motionlib_path = 'amass_data_unified_999_yup_motionlib'
    subdataset = ''#'CMU'
    file_list = [x for x in glob.glob(os.path.join(root_path, motionlib_path) + '/**/**/*_poses.npy') if subdataset in x]
    save_root = os.path.join(root_path, 'amass_with_babel')
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(root_path, motionlib_path, 'babel_all.json'), 'r') as f:
        babel_label = json.load(f)

    with open(os.path.join(root_path, motionlib_path, 'proc_label_abstract.json'), 'r') as f:
        label_abstract = json.load(f)

    cal_flag = 0
    motion_data = dict()
    motion_data['motions'] = list()
    for k in tqdm(babel_label):
        seq_dict = babel_label[k]
        seq_path = seq_dict['feat_p']
        seq_path = seq_path[seq_path.find('/'):].replace('.npz','.npy')
        full_path = os.path.join(root_path, motionlib_path) + seq_path

        if os.path.exists(full_path): #we ignored some of them when preprocessing
            curr_motion = SkeletonMotion.from_file(full_path)
            curr_fps = curr_motion.fps

            if seq_dict['frame_ann'] == None:
                seq = seq_dict['seq_ann']['labels'][0]
                act_cat = seq['act_cat']
                seq_id = seq['seg_id']
                if cal_flag:
                    curr_motion.to_file(os.path.join(save_root, seq_id + '.npy'))

                weight = get_weight_from_class(act_cat, label_abstract)
                motion_data['motions'].append({'file': seq_id + '.npy', 'weight': weight})

            else:
                frame_ann = seq_dict['frame_ann']['labels']
                for subseq in frame_ann:
                    act_cat = subseq['act_cat']
                    subseq_id = subseq['seg_id']

                    #to crop motion
                    start_frame = round(subseq['start_t'] * curr_fps)
                    end_frame = round(subseq['end_t'] * curr_fps)
                    if end_frame - start_frame < 3 or len(curr_motion) - start_frame < 3:
                        # print('too short')
                        continue
                    if cal_flag:
                        sub_motion = curr_motion.crop(start_frame, end_frame)
                        sub_motion.to_file(os.path.join(save_root, subseq_id + '.npy'))

                    weight = get_weight_from_class(act_cat, label_abstract)
                    motion_data['motions'].append({'file': subseq_id + '.npy', 'weight': weight})
        # break
    with open(os.path.join(save_root, 'motion_jumpstandup.yaml'), 'w') as f:
        f.write(yaml.dump(motion_data, indent=4))
    return

def main_getweight_pure():
    # load motion raw
    root_path = '/home/datassd/yuxuan'
    save_root = os.path.join(root_path, 'amass_with_babel_precomputed')
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'babel_all.json'), 'r') as f:
        babel_label = json.load(f)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proc_label_abstract.json'), 'r') as f:
        label_abstract = json.load(f)

    motion_file = '/home/datassd/yuxuan/amass_with_babel/motion_pretrain.yaml'
    motion_files = []

    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)

    motion_list = motion_config['motions']
    for motion_entry in motion_list:
        curr_file = motion_entry['file']
        seq_uid = curr_file[:-4]
        # print(curr_file)
        motion_files.append(seq_uid)
    print('motion_file_num:', len(motion_files))
    motion_data = dict()
    motion_data['motions'] = list()
    if not os.path.exists('walkrun_cmu_count.yaml'):
        simpledynamic_count = dict()
        for k in tqdm(babel_label):
            seq_dict = babel_label[k]
            if 'CMU' in seq_dict['feat_p']:
                if seq_dict['frame_ann'] == None:
                    if seq_dict['seq_ann']['mul_act'] == False:
                        seq = seq_dict['seq_ann']['labels'][0]
                        act_cat = seq['act_cat']
                        seq_id = seq['seg_id']
                        if seq_id in motion_files:
                            for act_c in act_cat:
                                if act_c in ['walk', 'run']:
                                    simpledynamic_count[act_c] = simpledynamic_count.get(act_c, 0) + 1
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'walkrun_cmu_count.yaml'), 'w') as f:
            f.write(yaml.dump(simpledynamic_count, indent=4))
    else:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'walkrun_cmu_count.yaml'), 'r') as f:
            simpledynamic_count = yaml.load(f, Loader=yaml.SafeLoader)
    print('motion_file_num walkrun_cmu_count:', sum(simpledynamic_count.values()))
    for k in tqdm(babel_label):
        seq_dict = babel_label[k]
        if seq_dict['frame_ann'] == None:
            seq = seq_dict['seq_ann']['labels'][0]
            act_cat = seq['act_cat']
            seq_id = seq['seg_id']
            weight = 0
            if seq_id in motion_files:
                for act_c in act_cat:
                    weight += 1.0 / simpledynamic_count.get(act_c, np.inf)
                motion_data['motions'].append({'file': seq_id + '.npy', 'weight': weight})
    print('motion_file_num check:', len(motion_data['motions']))
    with open(os.path.join(save_root, 'motion_walkrun_cmu.yaml'), 'w') as f:
        f.write(yaml.dump(motion_data, indent=4))
    return

def main_packdata(fold):
    motion_file = '/home/datassd/yuxuan/amass_with_babel/motion_pretrain.yaml'
    dir_name = os.path.dirname(motion_file)
    motion_files = []

    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)

    motion_list = motion_config['motions']
    for motion_entry in motion_list:
        curr_file = motion_entry['file']
        curr_weight = motion_entry['weight']
        assert(curr_weight >= 0)

        curr_file = os.path.join(dir_name, curr_file)
        motion_files.append(curr_file)

    length = len(motion_files)
    dividen = 5
    start = int(fold / dividen * length)
    end = int((fold+1) / dividen * length)
    motion_files_partial = motion_files[start: end]
    motion_pack = []
    for motion_path in tqdm(motion_files_partial):
        curr_motion = SkeletonMotion.from_file(motion_path)
        curr_motion = np.load(motion_path, allow_pickle=True).item()
        motion_pack.append(curr_motion)
    np.save(os.path.join(dir_name+'_pack', 'motion_pack_{:0>4d}'.format(fold)), motion_pack)
    return

def main_precompute(fold):
    motion_file = '/home/datassd/yuxuan/amass_with_babel/motion_pretrain.yaml'
    dir_name = os.path.dirname(motion_file)
    motion_files = []
    foldername = 'amass_with_babel_precomputed_rglob'
    os.makedirs(dir_name.replace('amass_with_babel', foldername).replace('datassd', 'data'), exist_ok=True)
    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)

    motion_list = motion_config['motions']
    for motion_entry in motion_list:
        curr_file = motion_entry['file']
        curr_weight = motion_entry['weight']
        assert(curr_weight >= 0)

        curr_file = os.path.join(dir_name, curr_file)
        motion_files.append(curr_file)

    length = len(motion_files)
    dividen = 1
    start = int(fold / dividen * length)
    end = int((fold+1) / dividen * length)
    motion_files_partial = motion_files[start: end]
    
    for motion_path in tqdm(motion_files_partial):
        curr_motion = SkeletonMotion_dof.from_file(motion_path)
        motion_path = motion_path.replace('amass_with_babel', foldername)
        motion_path = motion_path.replace('datassd', 'data')
        curr_motion.to_file(motion_path)

    return

if __name__ == '__main__':
    # main_getweight()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fold', type=int)
    # args = parser.parse_args()
    # main_packdata(args.fold)
    # main_precompute(0)
    main_getweight_pure()
