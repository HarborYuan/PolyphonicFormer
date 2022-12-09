import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob,
                      build_activation_layer, build_norm_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, build_transformer_layer
from mmdet.utils import get_root_logger
from polyphonic.funcs.utils import tensor_mask2box
from polyphonic.funcs.depth_utils import depth_act


@HEADS.register_module()
class KernelUpdateHead(nn.Module):

    def __init__(
            self,
            num_classes=80,
            num_thing_classes=80,
            num_stuff_classes=53,
            num_ffn_fcs=2,
            num_heads=8,
            num_cls_fcs=1,
            num_mask_fcs=3,
            feedforward_channels=2048,
            in_channels=256,
            out_channels=256,
            dropout=0.0,
            mask_thr=0.5,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_act_cfg=dict(type='ReLU', inplace=True),
            conv_kernel_size=3,
            feat_transform_cfg=None,
            hard_mask_thr=0.5,
            kernel_init=False,
            with_ffn=True,
            mask_out_stride=4,
            relative_coors=False,
            relative_coors_off=False,
            feat_gather_stride=1,
            mask_transform_stride=1,
            mask_upsample_stride=1,
            mask_assign_stride=4,
            ignore_label=255,
            kernel_updator_cfg=dict(
                type='DynamicConv',
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=1,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),
            loss_rank=None,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            loss_dice=dict(type='DiceLoss', loss_weight=3.0),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_depth=dict(
                type='DepthLoss',
                loss_weight=1.0,
                act=True,
                si_weight=1.0,
                sq_rel_weight=1.0,
                abs_rel_weight=1.0,
            ),
            depth_act_mode='monodepth'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.loss_depth = build_loss(loss_depth)
        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label

        self.attention = MultiheadAttention(
            in_channels * conv_kernel_size ** 2, num_heads, dropout)
        self.attention_depth = MultiheadAttention(
            in_channels * conv_kernel_size ** 2, num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size ** 2)[1]
        self.attention_norm_depth = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size ** 2)[1]

        self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)
        self.kernel_update_conv_depth = build_transformer_layer(kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
            self.feat_depth_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg
            )
        else:
            self.feat_transform = None
            self.feat_depth_transform = None

        if self.with_ffn:
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
            self.ffn_depth = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm_depth = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))

        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.depth_regs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.depth_regs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.depth_regs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])

        self.fc_mask = nn.Linear(in_channels, out_channels)
        self.fc_depth = nn.Linear(in_channels, out_channels)
        self.depth_act_mode = depth_act_mode

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.kernel_init:
            logger = get_root_logger()
            logger.info(
                'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)

    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None,
                depth_preds=None,
                depth_proposal=None,
                depth_feats=None):

        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
            depth_feats = self.feat_depth_transform(depth_feats)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)
        depth_feats_masked = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, depth_feats)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
        depth_proposal = depth_proposal.reshape(N, num_proposals, self.in_channels,
                                                -1).permute(0, 1, 3, 2)
        depth_proposal = depth_proposal + proposal_feat.detach()

        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)
        depth_feat_new = self.kernel_update_conv_depth(depth_feats_masked, depth_proposal)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        depth_feat_new = depth_feat_new.reshape(N, num_proposals, -1).permute(1, 0, 2)

        obj_feat = self.attention_norm(self.attention(obj_feat))
        depth_feat_new = self.attention_norm_depth(self.attention_depth(depth_feat_new))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)
        depth_feat_new = depth_feat_new.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)
        depth_feat_new = depth_feat_new.reshape(N, num_proposals, -1, self.in_channels)

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))
            depth_feat_new = self.ffn_norm_depth(self.ffn_depth(depth_feat_new))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat
        depth_pred_new = depth_feat_new

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        for depth_layer in self.depth_regs:
            depth_pred_new = depth_layer(depth_pred_new)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)
        depth_pred_new = self.fc_depth(depth_pred_new).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2
                and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # comments from K-Net
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        depth_pred_new = depth_pred_new.reshape(N, num_proposals, C,
                                                self.conv_kernel_size,
                                                self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        new_depth_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))
            new_depth_preds.append(
                F.conv2d(
                    depth_feats[i:i + 1],
                    depth_pred_new[i],
                    padding=int(self.conv_kernel_size // 2)
                )
            )

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_depth_preds = torch.cat(new_depth_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        new_depth_preds = new_depth_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size), new_depth_preds, depth_feat_new.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)

    @force_fp32(apply_to=('cls_score', 'mask_pred'))
    def loss(self,
             object_feats,
             cls_score,
             mask_pred,
             depth_pred,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             depth_targets,
             depth_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos).clamp_(min=1.0)

        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        assert mask_pred.shape[0] == cls_score.shape[0]
        assert mask_pred.shape[1] == cls_score.shape[1]

        if depth_pred is not None:
            assert depth_targets is not None
            assert depth_weights is not None
            H_D, W_D = depth_targets.shape[-2:]
            losses['loss_depth'] = self.loss_depth(
                depth_pred.reshape(depth_targets.shape[0], H_D, W_D),
                depth_targets,
                depth_weights
            )

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score.view(num_preds, -1),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(
                    cls_score.view(num_preds, -1)[pos_inds], labels[pos_inds])
        if mask_pred is not None:
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
                    batch_mask_targets = mask_targets.view(
                        batch_size, -1, H, W).bool()
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j
                    losses['loss_rank'] = self.loss_rank(
                        mask_pred, rank_target, ignore_index=self.ignore_label)
            else:
                losses['loss_mask'] = mask_pred.sum() * 0
                losses['loss_dice'] = mask_pred.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_pred.sum() * 0

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
        label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples, H, W)
        mask_weights[..., gt_valid.bool()] = 1.
        pos_mask_targets = None
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            pos_mask_targets = pos_gt_mask
            mask_targets[pos_inds, ...] = pos_mask_targets

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_mask.new_full((self.num_stuff_classes,),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_weights = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_mask.device)
            sem_thing_weights = pos_mask.new_zeros(
                (self.num_stuff_classes, self.num_thing_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1
            sem_weights = sem_weights * gt_valid

            label_weights[:, self.num_thing_classes:] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

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
                assert pos_mask_targets is not None
                depth_targets[pos_inds, ...] = gt_depth.repeat(num_pos, 1, 1)
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                depth_weights[pos_inds, ...] = pos_weight * pos_gt_mask

            if num_neg > 0:
                depth_weights[neg_inds, ...] = 0.

            if gt_sem_cls is not None and gt_sem_seg is not None and len(gt_sem_cls > 0):
                # gt sem cls may not always exist
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                sem_inds_total = sem_inds + num_samples
                depth_targets[sem_inds_total, ...] = gt_depth
                depth_weights[sem_inds_total, ...] = gt_sem_seg * pos_weight

            if True:
                direct_weight = 1.0
                depth_targets[-1, ...] = gt_depth
                depth_weights[-1, ...] = direct_weight

            depth_weights = torch.mul(depth_weights, depth_valid)
        else:
            depth_targets = None
            depth_weights = None

        return labels, label_weights, mask_targets, mask_weights, depth_targets, depth_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None,
                    gt_depth=None):

        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        gt_valid = [res.valid_mask for res in sampling_results]
        if gt_depth is not None:
            pos_depth_list = [res.pos_depth for res in sampling_results]
            neg_depth_list = [res.neg_depth for res in sampling_results]
        else:
            pos_depth_list = [None] * len(pos_inds_list)
            neg_depth_list = [None] * len(neg_inds_list)
            gt_depth = [None] * len(pos_inds_list)

        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_sem_seg is None:
            gt_sem_seg = [None] * 2
            gt_sem_cls = [None] * 2

        labels, label_weights, mask_targets, mask_weights, depth_targets, depth_weights = multi_apply(
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
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
            if gt_depth[0] is not None:
                depth_targets = torch.cat(depth_targets, 0)
                depth_weights = torch.cat(depth_weights, 0)
            else:
                depth_targets = None
                depth_weights = None
        return labels, label_weights, mask_targets, mask_weights, depth_targets, depth_weights

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    def rescale_depth(self, depth, img_meta):
        h, w, _ = img_meta['img_shape']
        depth = depth_act(depth, self.depth_act_mode)
        depth = F.interpolate(
            depth[None],
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        depth = depth[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        depth = F.interpolate(
            depth,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return depth

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      test_cfg, img_meta):
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg.mask_thr
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores, depth_preds):
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        depth_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        depth_preds = depth_preds.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes from the
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bboxes[:, :4] = np.array(tensor_mask2box(mask_preds).clip(min=0))

        mask_preds = mask_preds.cpu().numpy()

        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
            depth_result[det_labels[idx]].append(depth_preds[idx])
        return bboxes, segm_result, depth_result
