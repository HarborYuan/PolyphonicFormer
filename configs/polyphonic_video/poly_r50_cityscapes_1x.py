_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/polyphonic_former.py',
    '../_base_/datasets/cityscapes_dvps.py',
]

load_from = 'https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/polyphonic_r50_image.pth'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            split='train',
            ref_sample_mode='random',
            ref_seq_index=[-2, -1, 1, 2],
            test_mode=False,
            with_depth=True,
        )),
)

model = dict(
    type="PolyphonicVideo",
    rpn_head=dict(
        loss_depth=dict(
            loss_weight=1.,
            depth_act_mode='sigmoid',
        ),
    ),
    roi_head=dict(
        tracking=True,
    ),
    track_head=dict(
        type='QuasiDenseMaskEmbedHeadGTMask',
        num_convs=4,
        num_fcs=1,
        embed_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
        loss_track_aux=dict(
            type='L2Loss',
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            loss_weight=1.0),
    ),
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.35,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=5,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(
            type='RoIAlign', output_size=7, sampling_ratio=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]
    ),
    track_train_cfg=dict(
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
            mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
        sampler=dict(type='MaskPseudoSampler'),
    )
)
