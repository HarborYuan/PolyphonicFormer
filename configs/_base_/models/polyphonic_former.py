num_stages = 3
num_proposals = 100
conv_kernel_size = 1

num_thing_classes = 8
num_stuff_classes = 11

model = dict(
    type='Polyphonic',
    num_thing_classes=num_thing_classes,
    num_stuff_classes=num_stuff_classes,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4
    ),
    rpn_head=dict(
        type='KernelHead',
        num_proposals=num_proposals,
        num_classes=num_thing_classes + num_stuff_classes,
        num_thing_classes=num_thing_classes,
        num_stuff_classes=num_stuff_classes,
        # channel configs
        in_channels=256,
        out_channels=256,
        num_heads=8,
        # layer num configs
        num_cls_fcs=1,
        num_seg_convs=1,
        num_loc_convs=1,
        conv_kernel_size=conv_kernel_size,
        with_depth=True,
        cat_stuff_mask=True,
        feat_downsample_stride=2,
        feat_refine_stride=1,
        feat_refine=False,
        use_binary=True,
        num_depth_convs=1,
        conv_normal_init=True,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        feat_transform_cfg=None,
        loss_rank=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.1),
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=4.0),
        loss_depth=dict(
            type='DepthLoss',
            loss_weight=5.0,
            depth_act_mode='sigmoid',
            si_weight=1.0,
            sq_rel_weight=1.0,
            abs_rel_weight=1.0,
        ),
        localization_fpn=dict(
            type='SemanticFPNWrapper',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True
            ),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            # use aux to pred depth
            num_aux_convs=2,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        ),
    ),

    roi_head=dict(
        type='KernelUpdateIterHead',
        num_stages=num_stages,
        assign_stages=num_stages,
        recursive=False,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        num_proposals=num_proposals,
        num_thing_classes=num_thing_classes,
        num_stuff_classes=num_stuff_classes,
        do_panoptic=True,
        merge_joint=True,
        mask_head=dict(
            type='KernelUpdateHead',
            num_thing_classes=num_thing_classes,
            num_stuff_classes=num_stuff_classes,
            num_classes=num_thing_classes + num_stuff_classes,
            num_ffn_fcs=2,
            num_heads=8,
            num_cls_fcs=1,
            num_mask_fcs=1,
            feedforward_channels=2048,
            in_channels=256,
            out_channels=256,
            dropout=0.0,
            mask_thr=0.5,
            conv_kernel_size=conv_kernel_size,
            mask_upsample_stride=2,
            ffn_act_cfg=dict(type='ReLU', inplace=True),
            with_ffn=True,
            feat_transform_cfg=dict(
                conv_cfg=dict(type='Conv2d'), act_cfg=None),
            kernel_updator_cfg=dict(
                type='KernelUpdator',
                in_channels=256,
                feat_channels=256,
                out_channels=256,
                input_feat_shape=3,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),
            loss_rank=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.1),
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            loss_dice=dict(
                type='DiceLoss', loss_weight=4.0),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_depth=dict(
                type='DepthLoss',
                loss_weight=5.0,
                depth_act_mode='sigmoid',
                si_weight=1.0,
                sq_rel_weight=1.0,
                abs_rel_weight=1.0,
            ),
            depth_act_mode='sigmoid'
        )
    ),

    # training and test cfg
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssignerWithDepth',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1.),
        rcnn=dict(
            assigner=dict(
                type='MaskHungarianAssignerWithDepth',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0,
                               pred_act=True),
                depth_cost=dict(
                    type='DepthCost', weight=0.,
                    loss_fn=dict(
                        type='DepthMatchLoss',
                        loss_weight=1.,
                    ),
                    depth_act_mode='sigmoid'
                )
            ),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1.
        )
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=num_proposals,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3
            )
        )
    )
)
