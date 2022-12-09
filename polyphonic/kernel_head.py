import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, normal_init)
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss, build_neck
from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger


@HEADS.register_module()
class KernelHead(nn.Module):
    """
        This head for init mask and kernel prediction
    """

    def __init__(
            self,
            num_proposals=100,
            num_classes=133,
            num_thing_classes=80,
            num_stuff_classes=53,
            in_channels=256,
            out_channels=256,
            num_heads=8,
            num_cls_fcs=1,
            num_seg_convs=1,
            num_loc_convs=1,
            att_dropout=False,
            localization_fpn=None,
            conv_kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            semantic_fpn=True,
            train_cfg=None,
            xavier_init_kernel=False,
            kernel_init_std=0.01,
            use_binary=False,
            proposal_feats_with_obj=False,
            loss_mask=None,
            loss_seg=None,
            loss_cls=None,
            loss_dice=None,
            loss_rank=None,
            loss_depth=None,
            feat_downsample_stride=1,
            feat_refine_stride=1,
            feat_refine=True,
            conv_normal_init=False,
            mask_out_stride=4,
            hard_target=False,
            ignore_label=255,
            cat_stuff_mask=False,
            # Depth related
            with_depth=True,
            num_depth_convs=1,
            # Stuff-direct related
            semantic_out_cfg=None,
            loss_semantic_seg=None,
            **kwargs
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.proposal_feats_with_obj = proposal_feats_with_obj
        self.sampling = False
        self.localization_fpn = build_neck(localization_fpn)
        self.semantic_fpn = semantic_fpn
        self.norm_cfg = norm_cfg
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.conv_kernel_size = conv_kernel_size
        self.xavier_init_kernel = xavier_init_kernel
        self.kernel_init_std = kernel_init_std
        self.feat_downsample_stride = feat_downsample_stride
        self.feat_refine_stride = feat_refine_stride
        self.conv_normal_init = conv_normal_init
        self.feat_refine = feat_refine
        self.num_loc_convs = num_loc_convs
        self.num_seg_convs = num_seg_convs
        self.use_binary = use_binary
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.ignore_label = ignore_label
        self.cat_stuff_mask = cat_stuff_mask

        self.with_depth = with_depth
        self.num_depth_convs = num_depth_convs

        # Semantic out related
        self.semantic_out_cfg = semantic_out_cfg

        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)
        else:
            self.loss_mask = loss_mask

        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = loss_dice

        if loss_seg is not None:
            self.loss_seg = build_loss(loss_seg)
        else:
            self.loss_seg = loss_seg
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = loss_cls

        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        if loss_depth is not None:
            self.loss_depth = build_loss(loss_depth)
        else:
            self.loss_depth = None

        if loss_semantic_seg is not None:
            self.loss_semantic_seg = build_loss(loss_semantic_seg)
        else:
            self.loss_semantic_seg = None

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='MaskPseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_kernels = nn.Conv2d(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias=False)  # (N, C, 1, 1) -> (N, C)

        if self.semantic_fpn:
            if self.loss_seg.use_sigmoid:
                self.conv_seg = nn.Conv2d(self.out_channels, self.num_classes, 1)
            else:
                self.conv_seg = nn.Conv2d(self.out_channels, self.num_classes + 1, 1)

        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,  # 2
                padding=1,
                norm_cfg=self.norm_cfg)
            self.seg_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,  # 2
                padding=1,
                norm_cfg=self.norm_cfg)

        self.loc_convs = nn.ModuleList()
        for i in range(self.num_loc_convs):
            self.loc_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.seg_convs = nn.ModuleList()
        for i in range(self.num_seg_convs):
            self.seg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.depth_convs = nn.ModuleList()
        for i in range(self.num_depth_convs):
            self.depth_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.conv_direct_depth = nn.Conv2d(self.out_channels, 1, 1)

        # Semantic_out related
        if self.semantic_out_cfg:
            self.semantic_aspp = build_neck(self.semantic_out_cfg)
            self.semantic_aspp_predict = nn.Conv2d(
                in_channels=self.semantic_aspp.out_feat_channels,
                out_channels=self.num_classes,
                kernel_size=1,
            )
        else:
            self.semantic_aspp = None

    def init_weights(self):
        self.localization_fpn.init_weights()

        if self.feat_downsample_stride > 1 and self.conv_normal_init:
            logger = get_root_logger()
            logger.info('Initialize convs in KPN head by normal std 0.01')
            for conv in [self.loc_convs, self.seg_convs]:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)

        if self.semantic_fpn:
            bias_seg = bias_init_with_prob(0.01)
            if self.loss_seg.use_sigmoid:
                normal_init(self.conv_seg, std=0.01, bias=bias_seg)
            else:
                normal_init(self.conv_seg, mean=0, std=0.01)
        if self.xavier_init_kernel:
            logger = get_root_logger()
            logger.info('Initialize kernels by xavier uniform')
            nn.init.xavier_uniform_(self.init_kernels.weight)
        else:
            logger = get_root_logger()
            logger.info(
                f'Initialize kernels by normal std: {self.kernel_init_std}')
            normal_init(self.init_kernels, mean=0, std=self.kernel_init_std)

    def _decode_init_proposals(self, img, img_metas, train_tracking=False):
        num_imgs = len(img_metas)

        localization_feats = self.localization_fpn(img)

        # thing branch
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats)
        if self.feat_downsample_stride > 1 and self.feat_refine:
            loc_feats = self.ins_downsample(loc_feats)

        # init kernel prediction
        mask_preds = self.init_kernels(loc_feats)  # init mask prediction

        # stuff branch
        if self.semantic_fpn:
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                semantic_feats = self.seg_downsample(semantic_feats)
        else:
            semantic_feats = None

        # depth branch
        if self.with_depth:
            if isinstance(localization_feats, list):
                depth_feats = localization_feats[2]
            else:
                raise NotImplementedError
            for conv in self.depth_convs:
                depth_feats = conv(depth_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                raise NotImplementedError
        else:
            depth_feats = None

        if depth_feats is not None:
            depth_pred = self.conv_direct_depth(depth_feats)
            direct_depth_kernel = self.conv_direct_depth.weight.clone()
            direct_depth_kernel = direct_depth_kernel[None].expand(
                num_imgs, *direct_depth_kernel.size()
            )
        else:
            depth_pred = None
            direct_depth_kernel = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats)
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone()
        proposal_feats = proposal_feats[None].expand(num_imgs, *proposal_feats.size())

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats

        if self.semantic_aspp:
            semantic_aspp_out = self.semantic_aspp(x_feats)
            semantic_aspp_out = self.semantic_aspp_predict(semantic_aspp_out)
        else:
            semantic_aspp_out = None

        if self.proposal_feats_with_obj:
            sigmoid_masks = mask_preds.sigmoid()
            nonzero_inds = sigmoid_masks > 0.5
            if self.use_binary:
                sigmoid_masks = nonzero_inds.float()
            else:
                sigmoid_masks = nonzero_inds.float() * sigmoid_masks
            obj_feats = torch.einsum('bnhw, bchw->bnc', sigmoid_masks, x_feats)

        cls_scores = None

        if self.proposal_feats_with_obj:  # default True
            proposal_feats = proposal_feats + obj_feats.view(
                num_imgs, self.num_proposals, self.out_channels, 1, 1)

        depth_proposal = direct_depth_kernel
        if self.cat_stuff_mask and not self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:self.num_classes]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.num_thing_classes:self.num_classes].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)  # (b, N_{st}+N_{th}, c)
            depth_proposal = depth_proposal.expand(-1, proposal_feats.shape[1], -1, -1, -1)

        # if self.cat_stuff_mask and train_tracking:
        #     mask_preds = torch.cat(
        #         [mask_preds, seg_preds[:, self.num_thing_classes:self.num_classes]], dim=1)
        #     stuff_kernels = self.conv_seg.weight[self.num_thing_classes:self.num_classes].clone()
        #     stuff_kernels = stuff_kernels[None].expand(num_imgs,
        #                                                *stuff_kernels.size())
        #     proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)  # (b, N_{st}+N_{th}, c)
        #     depth_proposal = depth_proposal.expand(-1, proposal_feats.shape[1], -1, -1, -1)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds, depth_feats, depth_proposal, depth_pred, semantic_aspp_out

    def forward_train(
            self,
            img,
            img_metas,
            gt_masks,
            gt_labels,
            gt_sem_seg=None,
            gt_sem_cls=None,
            gt_depth=None
    ):
        """Forward function in training stage."""
        num_imgs = len(img_metas)
        results = self._decode_init_proposals(img, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds,
         depth_feats, depth_proposal, depth_pred, semantic_aspp_out) = results
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            if seg_preds is not None:
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
            if depth_pred is not None:
                scaled_depth_pred_0 = F.interpolate(
                    depth_pred,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False
                )
                scaled_depth_pred = scaled_depth_pred_0.expand(
                    (-1, self.num_proposals + self.num_stuff_classes, -1, -1))
            if semantic_aspp_out is not None:
                scaled_semantic_aspp_out = F.interpolate(
                    semantic_aspp_out,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
            else:
                scaled_semantic_aspp_out = None
        else:
            scaled_mask_preds = mask_preds  # thing
            scaled_seg_preds = seg_preds  # stuff
            scaled_depth_pred_0 = depth_pred
            scaled_depth_pred = depth_pred.expand((-1, self.num_proposals + self.num_stuff_classes, -1, -1))

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks

        sampling_results = []
        if cls_scores is None:
            detached_cls_scores = [None] * num_imgs
        else:
            detached_cls_scores = cls_scores.detach()

        for i in range(num_imgs):
            valid_mask = torch.cat((gt_masks[i], gt_sem_seg[i]), dim=0).sum(dim=0).bool().float()
            assign_result = self.assigner.assign(scaled_mask_preds[i].detach(),
                                                 detached_cls_scores[i],
                                                 gt_masks[i], gt_labels[i],
                                                 img_metas[i],
                                                 depth_pred=scaled_depth_pred[i],
                                                 gt_depth=gt_depth[i],
                                                 gt_valid=valid_mask)
            sampling_result = self.sampler.sample(assign_result,
                                                  scaled_mask_preds[i],
                                                  gt_masks[i],
                                                  depth=scaled_depth_pred[i])
            sampling_result.valid_mask = valid_mask
            sampling_results.append(sampling_result)

        mask_targets = self.get_targets(
            sampling_results,
            gt_masks,
            self.train_cfg,
            True,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            gt_depth=gt_depth)

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds, scaled_depth_pred,
                           proposal_feats, scaled_semantic_aspp_out, *mask_targets)

        losses['depth_dense'] = self.loss_depth(
            pred=scaled_depth_pred_0,
            target=gt_depth,
            mask_weight=(gt_depth > 0.).float()
        )

        if self.cat_stuff_mask and self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:self.num_classes]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.num_thing_classes:self.num_classes].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)
            depth_proposal = depth_proposal.expand(-1, proposal_feats.shape[1], -1, -1, -1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores, \
               depth_feats, depth_proposal, depth_pred, semantic_aspp_out

    def loss(self,
             mask_pred,
             cls_scores,
             seg_preds,
             depth_pred,
             proposal_feats,
             semantic_aspp_out,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             seg_targets,
             depth_targets,
             depth_weights,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_preds = mask_pred.shape[0] * mask_pred.shape[1]

        if depth_pred is not None:
            assert depth_targets is not None
            assert depth_weights is not None
            H_D, W_D = depth_targets.shape[-2:]
            losses['loss_depth'] = self.loss_depth(
                depth_pred.reshape(depth_targets.shape[0], H_D, W_D),
                depth_targets,
                depth_weights
            )

        if cls_scores is not None:
            num_pos = pos_inds.sum().float()
            avg_factor = reduce_mean(num_pos)
            assert mask_pred.shape[0] == cls_scores.shape[0]
            assert mask_pred.shape[1] == cls_scores.shape[1]
            losses['loss_rpn_cls'] = self.loss_cls(
                cls_scores.view(num_preds, -1),
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['rpn_pos_acc'] = accuracy(
                cls_scores.view(num_preds, -1)[pos_inds], labels[pos_inds])

        bool_pos_inds = pos_inds.type(torch.bool)
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        # do not perform bounding box regression for BG anymore.
        H, W = mask_pred.shape[-2:]
        if pos_inds.any():
            pos_mask_pred = mask_pred.reshape(num_preds, H, W)[bool_pos_inds]
            pos_mask_targets = mask_targets[bool_pos_inds]
            pos_mask_weights = mask_weights[pos_inds].bool()
            losses['loss_rpn_mask'] = self.loss_mask(pos_mask_pred[pos_mask_weights],
                                                     pos_mask_targets[pos_mask_weights])
            loss_dice = []
            for i in range(pos_mask_pred.shape[0]):
                loss_dice.append(self.loss_dice(pos_mask_pred[i][pos_mask_weights[i]][None],
                                                pos_mask_targets[i][pos_mask_weights[i]][None]))
            losses['loss_rpn_dice'] = torch.stack(loss_dice).mean()

            if self.loss_rank is not None:
                batch_size = mask_pred.size(0)
                rank_target = mask_targets.new_full((batch_size, H, W),
                                                    self.ignore_label,
                                                    dtype=torch.long)
                rank_inds = pos_inds.view(batch_size,
                                          -1).nonzero(as_tuple=False)
                batch_mask_targets = mask_targets.view(batch_size, -1, H,
                                                       W).bool()
                for i in range(batch_size):
                    curr_inds = (rank_inds[:, 0] == i)
                    curr_rank = rank_inds[:, 1][curr_inds]
                    for j in curr_rank:
                        rank_target[i][batch_mask_targets[i][j]] = j
                losses['loss_rpn_rank'] = self.loss_rank(
                    mask_pred, rank_target, ignore_index=self.ignore_label)

        else:
            losses['loss_rpn_mask'] = mask_pred.sum() * 0
            losses['loss_rpn_dice'] = mask_pred.sum() * 0
            if self.loss_rank is not None:
                losses['loss_rank'] = mask_pred.sum() * 0

        if seg_preds is not None:
            if self.loss_seg.use_sigmoid:
                cls_channel = seg_preds.shape[1]
                pos_mask_weights = seg_targets != cls_channel
                flatten_seg = seg_preds.permute(1, 0, 2, 3)[..., pos_mask_weights].permute(1, 0)
                flatten_seg_target = seg_targets[pos_mask_weights]
                num_dense_pos = (flatten_seg_target >= 0) & (
                        flatten_seg_target < bg_class_ind)
                num_dense_pos = num_dense_pos.sum().float().clamp(min=1.0)
                losses['loss_rpn_seg'] = self.loss_seg(
                    flatten_seg,
                    flatten_seg_target,
                    avg_factor=num_dense_pos)
            else:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.view(-1, cls_channel, H * W).permute(
                    0, 2, 1).reshape(-1, cls_channel)
                flatten_seg_target = seg_targets.view(-1)
                losses['loss_rpn_seg'] = self.loss_seg(flatten_seg,
                                                       flatten_seg_target)
        if semantic_aspp_out is not None:
            cls_channel = semantic_aspp_out.shape[1]
            flatten_seg = semantic_aspp_out.view(-1, cls_channel, H * W).permute(
                0, 2, 1).reshape(-1, cls_channel)
            flatten_seg_target = seg_targets.view(-1)
            losses['loss_aspp_semseg'] = self.loss_semantic_seg(flatten_seg,
                                                                flatten_seg_target, ignore_index=self.num_classes)

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                           pos_depth, neg_depth, gt_depth, gt_valid,
                           cfg):
        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_mask.new_full((num_samples,),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_mask.new_zeros(num_samples)
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples, H, W)
        mask_weights[..., gt_valid.bool()] = 1.
        seg_targets = pos_mask.new_full((H, W),
                                        self.num_classes,
                                        dtype=torch.long)

        if gt_sem_cls is not None and gt_sem_seg is not None:
            gt_sem_seg = gt_sem_seg.bool()
            for sem_mask, sem_cls in zip(gt_sem_seg, gt_sem_cls):
                seg_targets[sem_mask] = sem_cls.long()

        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            mask_targets[pos_inds, ...] = pos_gt_mask
            for i in range(num_pos):
                seg_targets[pos_gt_mask[i].bool()] = pos_gt_labels[i]

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if pos_depth is not None:
            # Important : Be careful about depth weights since it may not compatible with depth losses
            assert neg_depth is not None
            assert gt_depth is not None
            # Here we use depth=0 in neg depth targets
            depth_targets = pos_depth.new_zeros((num_samples + self.num_stuff_classes, H, W))
            depth_weights = pos_depth.new_zeros((num_samples + self.num_stuff_classes, H, W))
            depth_valid = (gt_depth.repeat(num_samples + self.num_stuff_classes, 1, 1) > 0.).to(
                dtype=torch.float32).detach()
            if num_pos > 0:
                # when num_pos > 0, pos_mask_targets will not be None
                assert pos_gt_mask is not None
                depth_targets[pos_inds, ...] = gt_depth.repeat(num_pos, 1, 1)
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                depth_weights[pos_inds, ...] = pos_weight * pos_gt_mask

            if num_neg > 0:
                depth_weights[neg_inds, ...] = 0.

            if gt_sem_cls is not None and gt_sem_seg is not None and len(gt_sem_cls > 0):
                # gt sem cls may not always exist
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                sem_inds_total = gt_sem_cls - self.num_thing_classes + num_samples
                depth_targets[sem_inds_total, ...] = gt_depth
                depth_weights[sem_inds_total, ...] = gt_sem_seg * pos_weight

            depth_weights = torch.mul(depth_weights, depth_valid)
        else:
            depth_targets = None
            depth_weights = None

        return labels, label_weights, mask_targets, mask_weights, seg_targets, depth_targets, depth_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    rpn_train_cfg,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None,
                    gt_depth=None):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        gt_valid = [res.valid_mask for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        if gt_depth is not None:
            pos_depth_list = [res.pos_depth for res in sampling_results]
            neg_depth_list = [res.neg_depth for res in sampling_results]
        else:
            pos_depth_list = [None] * len(pos_inds_list)
            neg_depth_list = [None] * len(neg_inds_list)
            gt_depth = [None] * len(pos_inds_list)

        if gt_sem_seg is None:
            gt_sem_seg = [None] * 2
            gt_sem_cls = [None] * 2
        results = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            pos_depth_list,
            neg_depth_list,
            gt_depth,
            gt_valid,
            cfg=rpn_train_cfg)
        (labels, label_weights, mask_targets, mask_weights,
         seg_targets, depth_targets, depth_weights) = results
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
            seg_targets = torch.stack(seg_targets, 0)
            if gt_depth[0] is not None:
                depth_targets = torch.cat(depth_targets, 0)
                depth_weights = torch.cat(depth_weights, 0)
            else:
                depth_targets = None
                depth_weights = None
        return labels, label_weights, mask_targets, mask_weights, seg_targets, depth_targets, depth_weights

    def simple_test_rpn(self, img, img_metas, train_tracking=False):
        """Forward function in testing stage."""
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds,
         depth_feats, depth_proposal, depth_pred, semantic_aspp_out) = \
            self._decode_init_proposals(img, img_metas, train_tracking)
        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds, \
               depth_feats, depth_proposal, depth_pred, semantic_aspp_out

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)
