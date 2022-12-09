import numpy as np


def coords2bbox_all(coords):
    left = coords[:, 0].min().item()
    top = coords[:, 1].min().item()
    right = coords[:, 0].max().item()
    bottom = coords[:, 1].max().item()
    return top, left, bottom, right


def tensor_mask2box(masks):
    boxes = []
    for mask in masks:
        m = mask.nonzero().float()
        if m.numel() > 0:
            # box = coords2bbox(m, extend=2)
            box = coords2bbox_all(m)
        else:
            box = (-1, -1, 10, 10)
        boxes.append(box)
    return np.asarray(boxes)
