import numpy as np
import torch


def coords2bbox(coords, extend=2):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame
    """
    center = torch.mean(coords, dim=0)  # b * 2
    center = center.view(1, 2)
    center_repeat = center.repeat(coords.size(0), 1)

    dis_x = torch.sqrt(torch.pow(coords[:, 0] - center_repeat[:, 0], 2))
    dis_x = max(torch.mean(dis_x, dim=0).detach(), 1)
    dis_y = torch.sqrt(torch.pow(coords[:, 1] - center_repeat[:, 1], 2))
    dis_y = max(torch.mean(dis_y, dim=0).detach(), 1)

    left = center[:, 0] - dis_x * extend
    right = center[:, 0] + dis_x * extend
    top = center[:, 1] - dis_y * extend
    bottom = center[:, 1] + dis_y * extend

    return top.item(), left.item(), bottom.item(), right.item()


def mask2box(masks):
    boxes = []
    for mask in masks:
        m = mask[0].nonzero().float()
        if m.numel() > 0:
            box = coords2bbox(m, extend=2)
        else:
            box = (-1, -1, 10, 10)
        boxes.append(box)
    return np.asarray(boxes)


def coords2bboxTensor(coords, extend=2):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame
    """
    center = torch.mean(coords, dim=0)  # b * 2
    center = center.view(1, 2)
    center_repeat = center.repeat(coords.size(0), 1)

    dis_x = torch.sqrt(torch.pow(coords[:, 0] - center_repeat[:, 0], 2))
    dis_x = max(torch.mean(dis_x, dim=0).detach(), 1)
    dis_y = torch.sqrt(torch.pow(coords[:, 1] - center_repeat[:, 1], 2))
    dis_y = max(torch.mean(dis_y, dim=0).detach(), 1)

    left = center[:, 0] - dis_x * extend
    right = center[:, 0] + dis_x * extend
    top = center[:, 1] - dis_y * extend
    bottom = center[:, 1] + dis_y * extend

    return torch.Tensor([top.item(), left.item(), bottom.item(), right.item()]).to(coords.device)


def batch_mask2boxlist(masks):
    """
    Args:
        masks: Tensor b,n,h,w

    Returns: List[List[box]]

    """
    batch_bbox = []
    for i, b_masks in enumerate(masks):
        boxes = []
        for mask in b_masks:
            m = mask.nonzero().float()
            if m.numel() > 0:
                box = coords2bboxTensor(m, extend=2)
            else:
                box = torch.Tensor([0, 0, 0, 0]).to(m.device)
            boxes.append(box.unsqueeze(0))
        boxes_t = torch.cat(boxes, 0)
        batch_bbox.append(boxes_t)

    return batch_bbox


def bboxlist2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois
