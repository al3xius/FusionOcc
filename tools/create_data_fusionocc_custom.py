# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter_custom as nuscenes_converter

map_name_from_general_to_detection = {
    'noise': 'ignore',
    'animal': 'animal',
    'human': 'human',
    'movable_object': 'movable_object',
    'static_object': 'static_object',
    'static_object.rock': 'rock',
    'vehicle': 'vehicle',
    'surface.paved': 'paved',
    'surface.unpaved': 'unpaved',
    'surface.water': 'water',
    'vegetation.grass': 'grass',
    'vegetation.tree': 'tree',
    'vegetation.tree.canopy': 'tree_canopy',
    'vegetation.bush': 'bush',
    'vegetation.other': 'vegetation_other',
    'building': 'building',
    'bridge': 'bridge',
    'irrelevant': 'ignore'
}
classes = [
    'animal', 'human', 'movable_object', 'static_object', 'rock', 'vehicle',
    'paved', 'unpaved', 'water', 'grass', "tree", 'tree_canopy', 'bush', 'vegetation_other',
    'building', 'bridge', 'bridge'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/custom/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/custom/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/custom/gts/%s/%s'%(scene['name'], info['token'])
        with open('./data/custom/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'custom'
    version = 'v1.0'
    train_version = f'{version}-trainval'
    root_path = './data/custom'
    extra_tag = 'fusionocc-custom'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0   # 10
    )
    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
