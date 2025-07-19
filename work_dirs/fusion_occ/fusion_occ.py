point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageSeg',
        downsample=1,
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(512, 1408),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True,
        img_seg_dir='data/nuscenes/imgseg/samples'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='PointsLidar2Ego'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=True),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5])),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'points', 'sparse_depth', 'segs', 'voxel_semantics',
            'mask_camera'
        ])
]
test_pipeline = [
    dict(
        type='PrepareImageSeg',
        restore_upsample=8,
        downsample=1,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(512, 1408),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True,
        img_seg_dir='data/nuscenes/imgseg/samples'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='PointsLidar2Ego'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5])),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'points', 'sparse_depth'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='PrepareImageSeg',
                downsample=1,
                is_train=True,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(512, 1408),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True,
                img_seg_dir='data/nuscenes/imgseg/samples'),
            dict(type='LoadOccGTFromFile'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='FuseAdjacentSweeps',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='PointsLidar2Ego'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
            dict(
                type='LoadAnnotationsAll',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=True),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.4],
                    y=[-40, 40, 0.4],
                    z=[-1, 5.4, 0.4],
                    depth=[1.0, 45.0, 0.5])),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'points', 'sparse_depth', 'segs',
                    'voxel_semantics', 'mask_camera'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        use_mask=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=(1, 2, 1),
        multi_adj_frame_id_cfg_lidar=(1, 8, 1)),
    val=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageSeg',
                restore_upsample=8,
                downsample=1,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(512, 1408),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True,
                img_seg_dir='data/nuscenes/imgseg/samples'),
            dict(type='LoadOccGTFromFile'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='FuseAdjacentSweeps',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='PointsLidar2Ego'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
            dict(
                type='LoadAnnotationsAll',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.4],
                    y=[-40, 40, 0.4],
                    z=[-1, 5.4, 0.4],
                    depth=[1.0, 45.0, 0.5])),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'points', 'sparse_depth'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        use_mask=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=(1, 2, 1),
        multi_adj_frame_id_cfg_lidar=(1, 8, 1)),
    test=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageSeg',
                restore_upsample=8,
                downsample=1,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(512, 1408),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True,
                img_seg_dir='data/nuscenes/imgseg/samples'),
            dict(type='LoadOccGTFromFile'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='FuseAdjacentSweeps',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='PointsLidar2Ego'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
            dict(
                type='LoadAnnotationsAll',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.4],
                    y=[-40, 40, 0.4],
                    z=[-1, 5.4, 0.4],
                    depth=[1.0, 45.0, 0.5])),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'points', 'sparse_depth'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        use_mask=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=(1, 2, 1),
        multi_adj_frame_id_cfg_lidar=(1, 8, 1)))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='PrepareImageSeg',
            restore_upsample=8,
            downsample=1,
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(512, 1408),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            sequential=True,
            img_seg_dir='data/nuscenes/imgseg/samples'),
        dict(type='LoadOccGTFromFile'),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='FuseAdjacentSweeps',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(type='PointsLidar2Ego'),
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
        dict(
            type='LoadAnnotationsAll',
            bda_aug_conf=dict(
                rot_lim=(-0.0, 0.0),
                scale_lim=(1.0, 1.0),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='PointToMultiViewDepth',
            downsample=1,
            grid_config=dict(
                x=[-40, 40, 0.4],
                y=[-40, 40, 0.4],
                z=[-1, 5.4, 0.4],
                depth=[1.0, 45.0, 0.5])),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'points', 'sparse_depth'])
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='fusionocc',
                name='fusionocc-run-1',
                entity='al3xius-yt-technische-universit-t-graz'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fusion_occ'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(512, 1408),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 0.4],
    depth=[1.0, 45.0, 0.5])
use_mask = True
voxel_size = [0.05, 0.05, 0.05]
img_backbone_out_channel = 256
feature_channel = 32
lidar_out_channel = 32
img_channels = 32
numC_Trans = 64
num_classes = 18
multi_adj_frame_id_cfg = (1, 2, 1)
multi_adj_frame_id_cfg_lidar = (1, 8, 1)
model = dict(
    type='FusionOCC',
    lidar_in_channel=5,
    point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
    voxel_size=[0.05, 0.05, 0.05],
    lidar_out_channel=32,
    align_after_view_transformation=True,
    num_adj=1,
    fuse_loss_weight=0.1,
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=1536,
        out_channels=256,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='CrossModalLSS',
        feature_channels=32,
        seg_num_classes=18,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5]),
        input_size=(512, 1408),
        in_channels=256,
        mid_channels=128,
        depth_channels=88,
        is_train=True,
        out_channels=32,
        sid=False,
        collapse_z=False,
        depthnet_cfg=dict(aspp_mid_channels=96),
        downsample=16),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=32,
        with_cp=False,
        num_layer=[1],
        num_channels=[32],
        stride=[1],
        backbone_output_ids=[0]),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=96,
        num_layer=[1, 2, 3],
        with_cp=False,
        num_channels=[64, 128, 256],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    occ_encoder_neck=dict(type='LSSFPN3D', in_channels=448, out_channels=64),
    out_dim=64,
    loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    use_mask=True)
img_seg_dir = 'data/nuscenes/imgseg/samples'
bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
share_data_config = dict(
    use_mask=True,
    type='NuScenesDatasetOccpancy',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='fusionocc',
    multi_adj_frame_id_cfg=(1, 2, 1),
    multi_adj_frame_id_cfg_lidar=(1, 8, 1))
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageSeg',
            restore_upsample=8,
            downsample=1,
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(512, 1408),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            sequential=True,
            img_seg_dir='data/nuscenes/imgseg/samples'),
        dict(type='LoadOccGTFromFile'),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='FuseAdjacentSweeps',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(type='PointsLidar2Ego'),
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[-40, -40, -1, 40, 40, 5.4]),
        dict(
            type='LoadAnnotationsAll',
            bda_aug_conf=dict(
                rot_lim=(-0.0, 0.0),
                scale_lim=(1.0, 1.0),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='PointToMultiViewDepth',
            downsample=1,
            grid_config=dict(
                x=[-40, 40, 0.4],
                y=[-40, 40, 0.4],
                z=[-1, 5.4, 0.4],
                depth=[1.0, 45.0, 0.5])),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'points', 'sparse_depth'])
            ])
    ],
    ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
    use_mask=True,
    type='NuScenesDatasetOccpancy',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='fusionocc',
    multi_adj_frame_id_cfg=(1, 2, 1),
    multi_adj_frame_id_cfg_lidar=(1, 8, 1))
key = 'test'
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL')
]
gpu_ids = [0]
