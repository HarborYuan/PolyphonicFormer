import copy
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead
from polyphonic.funcs.sampler import MaskPseudoSampler


@HEADS.register_module()
class KernelUpdateIterHead(BaseRoIHead, ABC):

    def __init__(
            self,
            num_stages=6,
            recursive=False,
            assign_stages=5,
            stage_loss_weights=(1, 1, 1, 1, 1, 1),
            do_panoptic=False,
            proposal_feature_channel=256,
            merge_cls_scores=False,
            post_assign=False,
            hard_target=False,
            merge_joint=True,
            num_proposals=100,
            num_thing_classes=80,
            num_stuff_classes=53,
            mask_assign_stride=4,
            ignore_label=255,
            tracking=False,
            mask_head=dict(
                type='KernelUpdateHead',
                num_classes=80,
                num_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                hidden_channels=256,
                dropout=0.0,
                roi_feat_size=7,
                ffn_act_cfg=dict(type='ReLU', inplace=True)),
            mask_out_stride=4,
            train_cfg=None,
            test_cfg=None,
            **kwargs
    ):
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.merge_cls_scores = merge_cls_scores
        self.recursive = recursive
        self.post_assign = post_assign
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.merge_joint = merge_joint
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.num_proposals = num_proposals
        self.ignore_label = ignore_label
        self.tracking = tracking
        super().__init__(
            mask_head=mask_head, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(
                    self.mask_sampler[stage], MaskPseudoSampler), \
                    'Sparse Mask only support `MaskPseudoSampler`'

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        raise NotImplementedError

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            if isinstance(self.train_cfg, dict):
                _train_cfg = [copy.deepcopy(self.train_cfg) for _ in range(self.num_stages)]
                self.train_cfg = _train_cfg
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                # self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler)  # context=self
                )

    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if self.recursive:
            for i in range(self.num_stages):
                self.mask_head[i] = self.mask_head[0]

    def _mask_forward(self, stage, x, object_feats, mask_preds, img_metas,
                      depth_preds, depth_proposal, depth_feats):
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats, depth_preds, depth_proposal = mask_head(
            x, object_feats, mask_preds, img_metas=img_metas,
            depth_preds=depth_preds, depth_proposal=depth_proposal, depth_feats=depth_feats)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
            scaled_depth_preds = F.interpolate(
                depth_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear'
            )
        else:
            scaled_mask_preds = mask_preds
            scaled_depth_preds = depth_preds
        mask_results = dict(
            cls_score=cls_score,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats,
            scaled_depth_preds=scaled_depth_preds,
            depth_preds=depth_preds,
            depth_proposal=depth_proposal
        )

        return mask_results

    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_depth=None,
                      depth_preds=None,
                      depth_feats=None,
                      depth_proposal=None,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_bboxes=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None,
                      ):
        num_imgs = len(img_metas)
        depth_preds = depth_preds.expand(-1, depth_proposal.shape[1], -1, -1)
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
            prev_depth_preds_scaled = F.interpolate(
                depth_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()
            prev_depth_preds_scaled = depth_preds.detach()

        if cls_score is not None:
            prev_cls_score = cls_score.detach()
        else:
            prev_cls_score = [None] * num_imgs

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        assign_results = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas,
                                              depth_preds, depth_proposal, depth_feats)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']
            # The depth
            depth_proposal = mask_results['depth_proposal']
            scaled_depth_preds = mask_results['scaled_depth_preds']
            depth_preds = mask_results['depth_preds']

            if self.post_assign:
                raise NotImplementedError
                # prev_mask_preds = scaled_mask_preds.detach()
                # prev_cls_score = cls_score.detach()

            sampling_results = []
            if stage < self.assign_stages:
                assign_results = []
            for i in range(num_imgs):
                if stage < self.assign_stages:
                    mask_for_assign = prev_mask_preds[i][:self.num_proposals]
                    depth_for_assign = prev_depth_preds_scaled[i][:self.num_proposals]
                    if prev_cls_score[i] is not None:
                        cls_for_assign = prev_cls_score[
                                             i][:self.num_proposals, :self.num_thing_classes]
                    else:
                        cls_for_assign = None

                    valid_mask = torch.cat((gt_masks[i], gt_sem_seg[i]), dim=0).sum(dim=0).bool().float()
                    assign_result = self.mask_assigner[stage].assign(
                        mask_for_assign, cls_for_assign, gt_masks[i],
                        gt_labels[i], img_metas[i],
                        depth_pred=depth_for_assign, gt_depth=gt_depth[i],
                        gt_valid=valid_mask)
                    assign_results.append(assign_result)
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], scaled_mask_preds[i], gt_masks[i],
                    depth=scaled_depth_preds[i])
                sampling_result.valid_mask = valid_mask
                sampling_results.append(sampling_result)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                gt_masks,
                gt_labels,
                self.train_cfg[stage],
                True,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls, gt_depth=gt_depth)

            single_stage_loss = self.mask_head[stage].loss(
                object_feats,
                cls_score,
                scaled_mask_preds,
                scaled_depth_preds,
                *mask_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f's{stage}_{key}'] = value * \
                                                    self.stage_loss_weights[stage]

            if not self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()
                prev_depth_preds_scaled = scaled_depth_preds.detach()

        if not self.tracking:
            return all_stage_loss
        else:
            return all_stage_loss, object_feats, cls_score, mask_preds, scaled_mask_preds

    def simple_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    depth_preds=None,
                    depth_feats=None,
                    depth_proposal=None,
                    imgs_whwh=None,
                    aspp_semantic=None,
                    rescale=False,
                    semantic_input=None):

        # Decode initial proposals
        num_imgs = len(img_metas)
        # num_proposals = proposal_feats.size(1)
        depth_inital = depth_preds.clone().detach()
        depth_preds = depth_preds.expand(-1, depth_proposal.shape[1], -1, -1)

        if self.mask_head[0].mask_upsample_stride > 1:
            depth_inital = F.interpolate(
                depth_inital,
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)

            if aspp_semantic is not None:
                aspp_semantic = F.interpolate(
                    aspp_semantic,
                    scale_factor=self.mask_head[0].mask_upsample_stride,
                    mode='bilinear',
                    align_corners=False)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas,
                                              depth_preds, depth_proposal, depth_feats)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            # The depth
            depth_proposal = mask_results['depth_proposal']
            scaled_depth_preds = mask_results['scaled_depth_preds']
            depth_preds = mask_results['depth_preds']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(
                    cls_score[img_id],
                    scaled_mask_preds[img_id],
                    self.test_cfg,
                    img_metas[img_id],
                    depth_preds=scaled_depth_preds[img_id],
                    depth_init=depth_inital[img_id],
                    aspp_semantic=aspp_semantic[img_id]
                    if aspp_semantic is not None else None,
                    semantic_input=semantic_input
                )
                results.append(single_result)
        else:
            raise NotImplementedError
        return results

    def simple_test_mask_preds(self,
                               x,
                               proposal_feats,
                               mask_preds,
                               cls_score,
                               img_metas,
                               depth_preds=None,
                               depth_feats=None,
                               depth_proposal=None,
                               imgs_whwh=None,
                               rescale=False
                               ):

        # Decode initial proposals
        num_imgs = len(img_metas)

        depth_inital = depth_preds.clone().detach()
        depth_preds = depth_preds.expand(-1, depth_proposal.shape[1], -1, -1)

        if self.mask_head[0].mask_upsample_stride > 1:
            depth_inital = F.interpolate(
                depth_inital,
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas,
                                              depth_preds, depth_proposal, depth_feats)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            # The depth
            depth_proposal = mask_results['depth_proposal']
            scaled_depth_preds = mask_results['scaled_depth_preds']
            depth_preds = mask_results['depth_preds']

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        return object_feats, cls_score, mask_preds, scaled_mask_preds

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('SparseMask does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        C, H, W = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, C, -1)).view(
            num_imgs, num_proposals, H, W)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

    def get_panoptic(self, cls_scores, mask_preds, test_cfg, img_meta,
                     depth_preds, depth_init, aspp_semantic, semantic_input=None):
        depth_pred = self.mask_head[-1].rescale_depth(depth_preds, img_meta)
        depth_init = self.mask_head[-1].rescale_depth(depth_init, img_meta)
        aspp_semantic = self.mask_head[-1].rescale_masks(aspp_semantic, img_meta) \
            if aspp_semantic is not None else None
        # resize mask predictions back
        thing_scores = cls_scores[:self.num_proposals][:, :self.num_thing_classes]
        thing_mask_preds = mask_preds[:self.num_proposals]
        thing_scores, topk_indices = thing_scores.flatten(0, 1).topk(
            self.test_cfg.max_per_img, sorted=True)
        mask_indices = topk_indices // self.num_thing_classes
        thing_labels = topk_indices % self.num_thing_classes
        masks_per_img = thing_mask_preds[mask_indices]
        thing_masks = self.mask_head[-1].rescale_masks(masks_per_img, img_meta)
        if not self.merge_joint:
            thing_masks = thing_masks > test_cfg.mask_thr

        depth_pred_things = depth_pred[:self.num_proposals]
        depth_pred_things = depth_pred_things[mask_indices]
        bbox_result, segm_result, depth_result_thing = self.mask_head[-1].segm2result(
            thing_masks, thing_labels, thing_scores, depth_pred_things)

        depth_pred_stuff = depth_pred[self.num_proposals:]
        depth_final = depth_init.squeeze(0)
        depth_basic = depth_final.clone()

        stuff_scores = cls_scores[
                       self.num_proposals:][:, self.num_thing_classes:].diag()
        stuff_scores, stuff_inds = torch.sort(stuff_scores, descending=True)
        stuff_masks = mask_preds[self.num_proposals:][stuff_inds]
        stuff_masks = self.mask_head[-1].rescale_masks(stuff_masks, img_meta)
        if not self.merge_joint:
            stuff_masks = stuff_masks > test_cfg.mask_thr

        depth_pred_stuff = depth_pred_stuff[stuff_inds]

        if self.merge_joint:
            stuff_labels = stuff_inds + self.num_thing_classes
            panoptic_result = self.merge_stuff_thing_stuff_joint(thing_masks, thing_labels,
                                                                 thing_scores, stuff_masks,
                                                                 stuff_labels, stuff_scores,
                                                                 test_cfg.merge_stuff_thing,
                                                                 depth_final, depth_pred_things,
                                                                 depth_pred_stuff
                                                                 )
        else:
            raise NotImplementedError
        return None, None, panoptic_result, depth_basic.cpu().numpy(), depth_final.cpu().numpy()

    def merge_stuff_thing_stuff_joint(self,
                                      thing_masks,
                                      thing_labels,
                                      thing_scores,
                                      stuff_masks,
                                      stuff_labels,
                                      stuff_scores,
                                      merge_cfg=None,
                                      depth_all=None,
                                      depth_things=None,
                                      depth_stuff=None,
                                      ):

        H, W = thing_masks.shape[-2:]
        panoptic_seg = thing_masks.new_zeros((H, W), dtype=torch.int32)

        total_masks = torch.cat([thing_masks, stuff_masks], dim=0)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)
        total_depth = torch.cat([depth_things, depth_stuff], dim=0)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        segments_info = []
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg.overlap_thr:
                    continue
                current_segment_id += 1

                panoptic_seg[mask] = current_segment_id
                if depth_all is not None:
                    depth_all[mask] = total_depth[k][mask]

                if isthing:
                    segments_info.append({
                        'id': current_segment_id,
                        'isthing': isthing,
                        'score': total_scores[k].item(),
                        'category_id': pred_class,
                        'instance_id': k.item(),
                    })
                else:
                    segments_info.append({
                        'id': current_segment_id,
                        'isthing': isthing,
                        'category_id': pred_class,
                        'area': mask_area,
                    })

        return panoptic_seg.cpu().numpy(), segments_info
