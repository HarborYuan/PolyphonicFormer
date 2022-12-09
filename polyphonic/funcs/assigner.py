import numpy as np
import torch
from mmdet.core import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost
from torch import nn

from .depth_utils import depth_act

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@MATCH_COST.register_module()
class DepthMatchLoss(nn.Module):
    # only do loss
    def __init__(
            self,
            loss_weight=1.,
            loss_si=1.,
            loss_sq_rel=1.,
            loss_abs_rel=1.,
    ):
        self.loss_weight = loss_weight
        self.loss_si = loss_si
        self.loss_sq_rel = loss_sq_rel
        self.loss_abs_rel = loss_abs_rel
        self.eps = 1.e-5

    def __call__(self, inputs, targets, num_valid):
        inputs += self.eps
        targets += self.eps
        log_minus = torch.log(inputs) - torch.log(targets)
        minus = inputs - targets
        # Scale Invariant error
        si_err = torch.sum(torch.square(log_minus), dim=(-1, -2)) / num_valid - \
                 torch.sum(log_minus, dim=(-1, -2)) / torch.square(num_valid)
        # Square relative error
        sq_rel_err = torch.sqrt(torch.sum(torch.square(minus / targets), dim=(-1, -2)) / num_valid)
        # Absolute relative error
        abs_rel_err = torch.sum(torch.abs(minus / targets), dim=(-1, -2)) / num_valid

        return self.loss_weight * (
                self.loss_si * si_err + self.loss_sq_rel * sq_rel_err + self.loss_abs_rel * abs_rel_err)


@MATCH_COST.register_module()
class DepthCost(object):
    def __init__(
            self,
            weight=1.,
            loss_fn=dict(
                type='DepthMatchLoss',
                loss_weight=1.,
            ),
            depth_act_mode='monodepth'
    ):
        self.weight = weight
        self.loss_fn = build_match_cost(loss_fn)
        self.depth_act_mode = depth_act_mode

    def __call__(self, inputs, depth_gt, target_masks):
        n = inputs.shape[0]
        m = target_masks.shape[0]

        inputs = depth_act(inputs, mode=self.depth_act_mode)

        tgt_masked = torch.mul(depth_gt.repeat(m, 1, 1), target_masks)

        inputs_repeat = inputs.repeat(m, 1, 1, 1).permute((1, 0, 2, 3))
        tgt_masked_repeat = tgt_masked.repeat(n, 1, 1, 1)

        valid_mask = (tgt_masked_repeat > 0.).to(dtype=torch.float32).detach()
        input_masked_repeat = torch.mul(inputs_repeat, valid_mask)
        valid_num = valid_mask.sum(dim=(-1, -2)).clamp(min=0.001)
        loss = self.loss_fn(input_masked_repeat, tgt_masked_repeat, valid_num)
        # loss will be equal to 100 when no valid points, it is a explicit penalty for no valid points
        return loss * self.weight


@MATCH_COST.register_module()
class DiceCost(object):
    """DiceCost.

     Args:
         weight (int | float, optional): loss_weight
         pred_act (bool): Whether to activate the prediction
            before calculating cost

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight=1.,
                 pred_act=False,
                 act_mode='sigmoid',
                 eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode
        self.eps = eps

    def dice_loss(cls, input, target, valid, eps=1e-3):
        input = input.reshape(input.size()[0], -1)
        target = target.reshape(target.size()[0], -1).float()
        if valid is not None:
            valid = valid.reshape(-1)
            # einsum saves 10x memory
            # a = torch.sum(input[:, None] * target[None, ...], -1)
            a = torch.einsum('nh,mh,h->nm', input, target, valid)
            b = torch.sum(input * input * valid, 1) + eps
            c = torch.sum(target * target * valid, 1) + eps
        else:
            a = torch.einsum('nh,mh->nm', input, target)
            b = torch.sum(input * input, 1) + eps
            c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b[:, None] + c[None, ...])
        # 1 is a constance that will not affect the matching, so ommitted
        return -d

    def __call__(self, mask_preds, gt_masks, gt_valid=None):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            mask_preds = mask_preds.sigmoid()
        elif self.pred_act:
            mask_preds = mask_preds.softmax(dim=0)
        dice_cost = self.dice_loss(mask_preds, gt_masks, gt_valid, self.eps)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class MaskCost(object):
    """MaskCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_act=False, act_mode='sigmoid'):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode

    def __call__(self, cls_pred, target, gt_valid=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            cls_pred = cls_pred.sigmoid()
        elif self.pred_act:
            cls_pred = cls_pred.softmax(dim=0)
        num_proposals = cls_pred.shape[0]
        num_gts, H, W = target.shape
        # flatten_cls_pred = cls_pred.view(num_proposals, -1)
        # eingum is ~10 times faster than matmul
        if gt_valid is not None:
            pos_cost = torch.einsum('nhw,mhw,hw->nm', cls_pred, target, gt_valid)
            neg_cost = torch.einsum('nhw,mhw,hw->nm', 1 - cls_pred, 1 - target, gt_valid)
            # flatten_target = target.view(num_gts, -1).t()
            # pos_cost = flatten_cls_pred.matmul(flatten_target)
            # neg_cost = (1 - flatten_cls_pred).matmul(1 - flatten_target)
            cls_cost = -(pos_cost + neg_cost) / gt_valid.sum()
        else:
            pos_cost = torch.einsum('nhw,mhw->nm', cls_pred, target)
            neg_cost = torch.einsum('nhw,mhw->nm', 1 - cls_pred, 1 - target)
            cls_cost = -(pos_cost + neg_cost) / (H * W)

        return cls_cost * self.weight


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 mask_cost=dict(type='SigmoidCost', weight=1.0),
                 dice_cost=dict(),
                 boundary_cost=None,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None
        self.topk = topk

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_pids=None,
               img_meta=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        assigned_instance_ids = bbox_pred.new_full((num_bboxes,),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        if self.mask_cost.weight != 0:
            reg_cost = self.mask_cost(bbox_pred, gt_bboxes)
        else:
            reg_cost = 0
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(bbox_pred, gt_bboxes)
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(bbox_pred, gt_bboxes)
        else:
            b_cost = 0
        cost = cls_cost + reg_cost + dice_cost + b_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        if gt_pids is not None:
            assigned_instance_ids[matched_row_inds] = gt_pids[matched_col_inds]
            results = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            results.set_extra_property("pids", assigned_instance_ids)
            return results
        else:
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssignerWithDepth(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 mask_cost=dict(type='SigmoidCost', weight=1.0),
                 dice_cost=dict(),
                 depth_cost=None,
                 boundary_cost=None,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None

        if depth_cost is not None:
            self.depth_cost = build_match_cost(depth_cost)
        else:
            self.depth_cost = None
        self.topk = topk

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta=None,
               gt_bboxes_ignore=None,
               depth_pred=None,
               gt_depth=None,
               gt_valid=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

            gt_depth is a single channel map
            depth_pred is per-label maps

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        if self.mask_cost.weight != 0:
            reg_cost = self.mask_cost(bbox_pred, gt_bboxes, gt_valid)
        else:
            reg_cost = 0
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(bbox_pred, gt_bboxes, gt_valid)
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(bbox_pred, gt_bboxes)
        else:
            b_cost = 0

        # now calculate depth cost
        if self.depth_cost is not None and self.depth_cost.weight != 0 and \
                gt_depth is not None and depth_pred is not None:
            depth_cost = self.depth_cost(inputs=depth_pred, depth_gt=gt_depth, target_masks=gt_bboxes)
        else:
            depth_cost = 0.

        cost = cls_cost + reg_cost + dice_cost + b_cost + depth_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
