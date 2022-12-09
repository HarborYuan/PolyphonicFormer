import numpy as np

import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector


@DETECTORS.register_module()
class Polyphonic(TwoStageDetector):

    def __init__(
            self,
            *args,
            num_thing_classes=80,
            num_stuff_classes=53,
            mask_assign_stride: int = 4,
            semantic_kitti=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.with_rpn, 'KNet does not support external proposals'

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.semantic_kitti = semantic_kitti

    def forward_train(
            self,
            img,
            img_metas,
            gt_bboxes=None,
            gt_labels=None,
            gt_bboxes_ignore=None,
            gt_masks=None,
            gt_semantic_seg=None,
            gt_depth=None,
            ref_img=None,
            ref_img_metas=None,
            ref_gt_bboxes=None,
            ref_gt_bboxes_ignore=None,
            ref_gt_labels=None,
            ref_gt_masks=None,
            ref_gt_semantic_seg=None,
            ref_gt_depth=None,
            proposals=None,
            **kwargs
    ):
        """
        No docs now.
        """
        super(TwoStageDetector, self).forward_train(img, img_metas)
        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None
        # gt_masks and gt_semantic_seg are not padded when forming batch
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        _gt_masks = gt_masks
        _gt_labels = gt_labels

        gt_masks = []
        gt_labels = []
        gt_sem_seg = []
        gt_sem_cls = []
        for idx, (gt_mask, gt_label) in enumerate(zip(_gt_masks, _gt_labels)):
            mask_tensor = gt_mask.to_tensor(torch.float32, _gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0), (assign_H, assign_W),
                mode='bilinear' if not self.semantic_kitti else 'nearest',
                align_corners=False if not self.semantic_kitti else None).squeeze(0)

            thing_masks = mask_tensor[gt_label < self.num_thing_classes]
            thing_labels = gt_label[gt_label < self.num_thing_classes]
            gt_masks.append(thing_masks)
            gt_labels.append(thing_labels)
            stuff_masks = mask_tensor[gt_label >= self.num_thing_classes]
            stuff_labels = gt_label[gt_label >= self.num_thing_classes]
            gt_sem_seg.append(stuff_masks)
            gt_sem_cls.append(stuff_labels)

        gt_depth = F.interpolate(
            gt_depth, (assign_H, assign_W),
            mode='nearest',
        )

        x = self.extract_feat(img)
        rpn_results = self.rpn_head.forward_train(
            img=x,
            img_metas=img_metas,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            gt_depth=gt_depth
        )
        (rpn_losses, proposal_feats, x_feats, mask_preds, cls_scores,
         depth_feats, depth_proposal, depth_pred, semantic_aspp_out) = rpn_results

        losses = self.roi_head.forward_train(
            x=x_feats,
            proposal_feats=proposal_feats,
            mask_preds=mask_preds,
            cls_score=cls_scores,
            img_metas=img_metas,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            gt_depth=gt_depth,
            depth_preds=depth_pred,
            depth_feats=depth_feats,
            depth_proposal=depth_proposal,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.
            proposals : I don't know.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds, depth_feats, depth_proposal, depth_pred, semantic_aspp_out) = rpn_results
        segm_results = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            depth_preds=depth_pred,
            depth_feats=depth_feats,
            depth_proposal=depth_proposal,
            imgs_whwh=None,
            aspp_semantic=semantic_aspp_out,
            rescale=rescale)
        return segm_results
