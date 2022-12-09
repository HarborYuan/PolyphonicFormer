import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

from polyphonic.funcs.depth_utils import depth_act


@weighted_loss
def depth_loss(inputs, targets, min_depth=0., max_depth=80.):
    targets, mask_weight = targets
    mask = (targets > min_depth) & (targets < max_depth) & (mask_weight != 0)
    if not torch.any(mask):
        return torch.zeros((3,), device=inputs.device, dtype=inputs.dtype)
    inputs = inputs[mask]
    targets = targets[mask]
    num_points = inputs.shape[0]

    log_minus = (torch.log(inputs) - torch.log(targets)) * mask_weight[mask]
    minus = (inputs - targets) * mask_weight[mask]
    # Scale Invariant error
    si_err = torch.sum(torch.square(log_minus)) / num_points - torch.sum(log_minus) / (num_points ** 2)
    # Square relative error
    sq_rel_err = torch.sqrt(torch.sum(torch.square(minus / targets)) / num_points)
    # Absolute relative error
    abs_rel_err = torch.sum(torch.abs(minus / targets)) / num_points

    return torch.stack((si_err, sq_rel_err, abs_rel_err))


@LOSSES.register_module()
class DepthLoss(nn.Module):

    def __init__(
            self,
            loss_weight=1.0,
            depth_act_mode='monodepth',
            si_weight=1.0,
            sq_rel_weight=1.0,
            abs_rel_weight=1.0
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.depth_act_mode = depth_act_mode
        self.weight = torch.tensor([si_weight, sq_rel_weight, abs_rel_weight], dtype=torch.float32)

    def forward(self,
                pred,
                target,
                mask_weight,
                reduction_override='mean',
                **kwargs):
        if self.weight is not None and not torch.any(self.weight > 0):
            return (pred * self.weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')

        pred = depth_act(pred, mode=self.depth_act_mode)

        loss = self.loss_weight * depth_loss(
            pred,
            (target, mask_weight),
            weight=self.weight.to(device=pred.device),
            reduction=reduction_override,
            **kwargs)
        return loss
