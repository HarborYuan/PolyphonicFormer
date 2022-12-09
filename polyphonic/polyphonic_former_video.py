import numpy as np

import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

from mmdet.models.builder import build_head, build_roi_extractor
from mmdet.core import build_assigner, build_sampler
from polyphonic.funcs.utils import tensor_mask2box
from polyphonic.video.qdtrack.builder import build_tracker
from polyphonic.video.utils import batch_mask2boxlist, bboxlist2roi


@DETECTORS.register_module()
class PolyphonicVideo(TwoStageDetector):

    def __init__(
            self,
            *args,
            num_thing_classes=80,
            num_stuff_classes=53,
            mask_assign_stride: int = 4,
            semantic_kitti=False,
            # video configs
            track_head=None,
            bbox_roi_extractor=None,
            track_train_cfg=None,
            tracker=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.with_rpn, 'KNet does not support external proposals'

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.semantic_kitti = semantic_kitti

        # tracker settings
        self.num_proposals = self.rpn_head.num_proposals

        self.tracker = None
        self.track_roi_assigner = None
        self.track_roi_sampler = None
        self.cnt = -1

        if track_head is not None:
            self.track_train_cfg = track_train_cfg
            self.track_head = build_head(track_head)
            self.init_track_assigner_sampler()
            self.track_roi_extractor = build_roi_extractor(
                bbox_roi_extractor)

        if tracker is not None:
            self.tracker_cfg = tracker

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)
        self.cnt = 1

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.track_roi_assigner = build_assigner(
            self.track_train_cfg.assigner)
        self.track_roi_sampler = build_sampler(
            self.track_train_cfg.sampler, context=self)

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
            # for tracking
            gt_instance_ids=None,
            ref_gt_instance_ids=None,
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
            gt_instance_ids[idx] = gt_instance_ids[idx][gt_label < self.num_thing_classes]
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

        num_ref_frames = ref_img.size(1)
        assert num_ref_frames == 1, "PolyphonicFormer uses two frames (1 img 1 ref_img) for training"
        # remove frame_num since they are all 0
        ref_img = ref_img.squeeze(1)
        ref_gt_semantic_seg = ref_gt_semantic_seg.squeeze(1)
        ref_gt_depth = ref_gt_depth.squeeze(1)
        ref_img_metas = [itm[0] for itm in ref_img_metas]
        ref_gt_bboxes = [itm[0] for itm in ref_gt_bboxes]
        ref_gt_masks = [itm[0] for itm in ref_gt_masks]
        ref_gt_labels = [itm[:, 1].to(dtype=torch.int64) for itm in ref_gt_labels]
        ref_gt_instance_ids = [itm[:, 1].to(dtype=torch.int64) for itm in ref_gt_instance_ids]
        assert img.size()[-2:] == ref_img.size()[-2:], "img and ref_img batch_shape should be same"
        _ref_gt_masks = ref_gt_masks
        _ref_gt_labels = ref_gt_labels

        ref_gt_masks = []
        ref_gt_labels = []
        ref_gt_sem_seg = []
        ref_gt_sem_cls = []
        for idx, (gt_mask, gt_label) in enumerate(zip(_ref_gt_masks, _ref_gt_labels)):
            mask_tensor = gt_mask.to_tensor(torch.float32, _ref_gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0), (assign_H, assign_W),
                mode='bilinear' if not self.semantic_kitti else 'nearest',
                align_corners=False if not self.semantic_kitti else None).squeeze(0)

            thing_masks = mask_tensor[gt_label < self.num_thing_classes]
            thing_labels = gt_label[gt_label < self.num_thing_classes]
            ref_gt_instance_ids[idx] = ref_gt_instance_ids[idx][gt_label < self.num_thing_classes]
            ref_gt_masks.append(thing_masks)
            ref_gt_labels.append(thing_labels)
            stuff_masks = mask_tensor[gt_label >= self.num_thing_classes]
            stuff_labels = gt_label[gt_label >= self.num_thing_classes]
            ref_gt_sem_seg.append(stuff_masks)
            ref_gt_sem_cls.append(stuff_labels)

        ref_gt_depth = F.interpolate(
            ref_gt_depth, (assign_H, assign_W),
            mode='nearest',
        )

        x = self.extract_feat(img)
        self.backbone.eval()
        self.neck.eval()
        with torch.no_grad():
            x_ref = self.extract_feat(ref_img)
        self.backbone.train()
        self.neck.train()

        # rpn for main frame
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

        # rpn for reference frame
        self.rpn_head.eval()
        ref_rpn_results = self.rpn_head.simple_test_rpn(x_ref, ref_img_metas)
        self.rpn_head.train()
        (ref_proposal_feats, ref_x_feats, ref_mask_preds, ref_cls_scores, ref_seg_preds,
         ref_depth_feats, ref_depth_proposal, ref_depth_pred, ref_semantic_aspp_out) = ref_rpn_results

        losses_roi, object_feats, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.forward_train(
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

        self.roi_head.eval()
        _, ref_cls_scores, ref_mask_preds, ref_scaled_mask_preds = self.roi_head.simple_test_mask_preds(
            ref_x_feats,
            ref_proposal_feats,
            ref_mask_preds,
            ref_cls_scores,
            ref_img_metas,
            depth_preds=ref_depth_pred,
            depth_feats=ref_depth_feats,
            depth_proposal=ref_depth_proposal,
            imgs_whwh=None,
        )
        self.roi_head.train()

        # ===== Tracking Part -==== #
        gt_match_indices = []
        for i in range(len(ref_gt_instance_ids)):
            ref_ids = ref_gt_instance_ids[i].cpu().numpy().tolist()
            gt_ids = gt_instance_ids[i].cpu().numpy().tolist()
            gt_pids = [ref_ids.index(i) if i in ref_ids else -1 for i in gt_ids]
            gt_match_indices.append(torch.LongTensor([gt_pids]).to(img.device)[0])

        key_sampling_results, ref_sampling_results = [], []
        num_imgs = len(img_metas)
        for i in range(num_imgs):
            assign_result = self.track_roi_assigner.assign(
                scaled_mask_preds[i][:self.num_proposals].detach(),
                cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                gt_masks[i], gt_labels[i], img_meta=img_metas[i]
            )
            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                mask_preds[i].detach(),
                gt_masks[i]
            )
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_scaled_mask_preds[i][:self.num_proposals].detach(),
                ref_cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                ref_gt_masks[i],
                ref_gt_labels[i],
                img_meta=ref_img_metas[i]
            )
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_mask_preds[i].detach(),
                ref_gt_masks[i]
            )
            ref_sampling_results.append(ref_sampling_result)

        # roi feature embeddings
        key_masks = [res.pos_gt_masks for res in key_sampling_results]
        for idx in range(len(key_masks)):
            key_masks[idx] = F.interpolate(
                key_masks[idx].unsqueeze(0),
                size=(pad_H, pad_W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            key_masks[idx] = (key_masks[idx].sigmoid() > 0.5).float()

        key_feats = self._track_forward(x, key_masks)

        # reference roi feature embeddings
        ref_masks = [res.pos_gt_masks for res in ref_sampling_results]
        for idx in range(len(ref_masks)):
            ref_masks[idx] = F.interpolate(
                ref_masks[idx].unsqueeze(0),
                size=(pad_H, pad_W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            ref_masks[idx] = (ref_masks[idx].sigmoid() > 0.5).float()
        ref_feats = self._track_forward(x_ref, ref_masks)

        match_feats = self.track_head.match(
            key_feats,
            ref_feats,
            key_sampling_results,
            ref_sampling_results
        )

        asso_targets = self.track_head.get_track_targets(
            gt_match_indices,
            key_sampling_results,
            ref_sampling_results
        )
        loss_track = self.track_head.loss(*match_feats, *asso_targets)

        losses = {}
        losses.update(rpn_losses)
        losses.update(losses_roi)
        losses.update(loss_track)
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

        semantic_thing = 1.
        results = segm_results[0]
        _, _, panoptic_result, _, depth_final = results
        panoptic_seg, segments_info = panoptic_result

        things_index_for_tracking, things_labels_for_tracking, thing_masks_for_tracking, things_score_for_tracking = \
            self.get_things_id_for_tracking(panoptic_seg, segments_info)
        things_labels_for_tracking = torch.Tensor(things_labels_for_tracking).to(x_feats.device).to(dtype=torch.int64)
        if len(things_labels_for_tracking) > 0:
            things_bbox_for_tracking = torch.zeros(
                (len(things_score_for_tracking), 5),
                dtype=torch.float,
                device=x_feats.device
            )
            things_bbox_for_tracking[:, 4] = torch.tensor(
                things_score_for_tracking,
                device=things_bbox_for_tracking.device
            )
            thing_masks_for_tracking_final = []
            for mask in thing_masks_for_tracking:
                thing_masks_for_tracking_final.append(
                    torch.Tensor(mask).unsqueeze(0).to(device=x_feats.device, dtype=torch.float32)
                )
            thing_masks_for_tracking_final = torch.cat(thing_masks_for_tracking_final, 0)
            thing_masks_for_tracking = thing_masks_for_tracking_final
            thing_masks_for_tracking_with_input = thing_masks_for_tracking_final * semantic_thing
            track_feats = self._track_forward(x, thing_masks_for_tracking_with_input)
            things_bbox_for_tracking[:, :4] = torch.tensor(
                tensor_mask2box(thing_masks_for_tracking_with_input),
                device=things_bbox_for_tracking.device
            )
            assert self.cnt > 0
            bboxes, labels, ids = self.tracker.match(
                bboxes=things_bbox_for_tracking,
                labels=things_labels_for_tracking,
                track_feats=track_feats,
                frame_id=self.cnt
            )
            self.cnt += 1
            ids = ids + 1
            ids[ids == -1] = 0
        else:
            ids = []

        track_maps = self.generate_track_id_maps(ids, thing_masks_for_tracking, panoptic_seg)
        semantic_map = self.get_semantic_seg(panoptic_seg, segments_info)
        return [{"sem": semantic_map, "track": track_maps, "depth": depth_final}]

    # tracking function
    def _track_forward(self, x, mask_pred):
        """Track head forward function used in both training and testing.
        We use mask pooling to get the fine grain features"""
        if not self.training:
            mask_pred = [mask_pred]
        bbox_list = batch_mask2boxlist(mask_pred)
        track_rois = bboxlist2roi(bbox_list)
        track_rois = track_rois.clamp(min=0.0)
        track_feats = self.track_roi_extractor(x[:self.track_roi_extractor.num_inputs], track_rois)
        track_feats = self.track_head(track_feats)

        return track_feats

    @staticmethod
    def get_things_id_for_tracking(panoptic_seg, seg_infos):
        idxs = []
        labels = []
        masks = []
        score = []
        for segment in seg_infos:
            if segment['isthing']:
                thing_mask = panoptic_seg == segment["id"]
                masks.append(thing_mask)
                idxs.append(segment["instance_id"])
                labels.append(segment['category_id'])
                score.append(segment['score'])
        return idxs, labels, masks, score

    def get_semantic_seg(self, panoptic_seg, segments_info):
        semantic_seg = np.ones(panoptic_seg.shape, dtype=np.uint8) * self.num_thing_classes + self.num_stuff_classes
        for segment in segments_info:
            semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"]
        return semantic_seg

    @staticmethod
    def generate_track_id_maps(ids, masks, panopitc_seg_maps):
        final_id_maps = np.zeros(panopitc_seg_maps.shape)
        if len(ids) == 0:
            return final_id_maps
        masks = masks.bool()
        for i, id in enumerate(ids):
            mask = masks[i].cpu().numpy()
            final_id_maps[mask] = id
        return final_id_maps
