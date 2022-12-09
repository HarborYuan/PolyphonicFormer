import os
from functools import partial

import mmcv
import numpy as np
import torch

from datasets.utils import vpq_eval, INSTANCE_DIVISOR
from .utils import track_parallel_progress

_EPSILON = 1e-15


def evaluate_clip_single_core(func, pred_names, gt_names, depth_thr, num_classes, **kwargs):
    assert num_classes > 0
    pred = [torch.load(pred_name) for pred_name in pred_names]
    gt = [torch.load(gt_name) for gt_name in gt_names]

    pred_pan = [itm['panseg'] for itm in pred]
    gt_pan = [itm['panseg'] for itm in gt]
    pred_pan = np.concatenate(pred_pan, axis=1)
    gt_pan = np.concatenate(gt_pan, axis=1)

    pred_dep = [itm['depth'] for itm in pred]
    gt_dep = [itm['depth'] for itm in gt]
    pred_dep = np.concatenate(pred_dep, axis=1)
    gt_dep = np.concatenate(gt_dep, axis=1)

    if depth_thr > 0.:
        depth_mask = gt_dep > 0.
        masked_pq_pred = pred_pan[depth_mask]
        ignored_pred_mask = \
            (np.abs(pred_dep[depth_mask] - gt_dep[depth_mask]) / gt_dep[depth_mask]) > depth_thr
        masked_pq_pred[ignored_pred_mask] = num_classes * INSTANCE_DIVISOR
        pred_pan[depth_mask] = masked_pq_pred

    return func([pred_pan, gt_pan], **kwargs)


def video_evaluate(eval_dir, eval_metrics, num_classes=0, num_things=0):
    assert num_classes > 0
    assert num_things > 0
    gt_dir = os.path.join(eval_dir, 'gt')
    pred_dir = os.path.join(eval_dir, 'pred')

    gt_names = list(mmcv.scandir(gt_dir))
    gt_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), gt_names)))
    gt_dirs = list(map(lambda x: os.path.join(gt_dir, x), gt_names))

    pred_names = list(mmcv.scandir(pred_dir))
    pred_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), pred_names)))
    pred_dirs = list(map(lambda x: os.path.join(pred_dir, x), pred_names))

    print("There are totally {} frames.".format(len(pred_dirs)))

    for metric in eval_metrics:
        assert metric in ["DVPQ"]
        length = len(pred_dirs)
        if metric == 'DVPQ':
            windows = [1, 2, 3, 4]
            depth_thrs = [0, 0.5, 0.25, 0.1]
            for k in windows:
                for depth_thr in depth_thrs:
                    lambda_sym = 'inf' if depth_thr == 0 else depth_thr
                    print("Evaluating DVPQ: k={}; lambda={}".format(k, lambda_sym))
                    tasks = []
                    for idx in range(length):
                        if idx + k - 1 >= length:
                            break
                        seq_id = int(os.path.basename(pred_dirs[idx]).split('_')[0])
                        seq_id_last = int(os.path.basename(pred_dirs[idx + k - 1]).split('_')[0])
                        if seq_id != seq_id_last:
                            continue
                        all_pred = []
                        all_gt = []
                        for j in range(k):
                            pred_cur = pred_dirs[idx + j]
                            gt_cur = gt_dirs[idx + j]
                            all_pred.append(pred_cur)
                            all_gt.append(gt_cur)

                        # multi core version
                        func = partial(vpq_eval, num_classes=num_classes)
                        tasks.append((func, all_pred, all_gt, depth_thr, num_classes))
                    if len(tasks) == 0:
                        print("Video len too small.")
                        continue
                    # multi core version
                    results = track_parallel_progress(
                        evaluate_clip_single_core,
                        tasks=tasks,
                        nproc=128,
                    )
                    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)[:num_classes]
                    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)[:num_classes]
                    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)[:num_classes]
                    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)[:num_classes]

                    sq = iou_per_class / (tp_per_class + _EPSILON)
                    rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + _EPSILON)
                    pq = sq * rq
                    # remove nan
                    pq = np.nan_to_num(pq)
                    tpq = pq[:num_things]  # thing
                    spq = pq[num_things:]  # stuff
                    print(
                        'DVPQ : {:.3f} DVPQ_thing : {:.3f} DVPQ_stuff : {:.3f}'.format(
                            pq.mean() * 100,
                            tpq.mean() * 100,
                            spq.mean() * 100)
                    )
