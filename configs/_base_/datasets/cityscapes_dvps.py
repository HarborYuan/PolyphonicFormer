dataset_type = 'CityscapesDVPSDataset'
data_root = 'data/cityscapes-dvps'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)

train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', mode='direct', with_depth=True),
    dict(type='SeqResizeWithDepth', img_scale=(1024, 2048), ratio_range=[1.0, 2.0], keep_ratio=True),
    dict(type='SeqFlipWithDepth', flip_ratio=0.5),
    dict(type='SeqRandomCropWithDepth', crop_size=(1024, 2048), share_params=True, check_id_match=80000),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_depth', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadImgDirect'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img', 'img_id', 'seq_id'],
                 meta_keys=[
                     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                     'flip_direction', 'img_norm_cfg', 'ori_filename'
                 ]),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            ref_sample_mode='random',
            ref_seq_index=None,
            test_mode=False,
            with_depth=True,
            pipeline=train_pipeline
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='img',
        ref_seq_index=None,
        test_mode=True,
        with_depth=True,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='img',
        ref_seq_index=None,
        test_mode=True,
        with_depth=True,
        pipeline=test_pipeline
    )
)

