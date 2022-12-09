import os
import random
from collections import defaultdict
from typing import List

import torch
from typing_extensions import Literal

import copy

import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.utils import get_root_logger

from datasets.utils import SeqObj, vpq_eval, INSTANCE_DIVISOR, compute_errors

# The classes
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
)

THING_CLASSES = (
    'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
)
STUFF_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky'
)

PALETTE = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

NO_OBJ = 32
NO_OBJ_HB = 255
DIVISOR_PAN = 1000
NUM_THING = len(THING_CLASSES)
NUM_STUFF = len(STUFF_CLASSES)


def build_classes():
    classes = []
    for cls in THING_CLASSES:
        classes.append(cls)

    for cls in STUFF_CLASSES:
        classes.append(cls)
    assert len(classes) == len(CLASSES)
    return classes


def build_palette():
    palette = []
    for cls in THING_CLASSES:
        palette.append(PALETTE[CLASSES.index(cls)])

    for cls in STUFF_CLASSES:
        palette.append(PALETTE[CLASSES.index(cls)])

    assert len(palette) == len(CLASSES)
    return palette


def to_coco(pan_map, divisor=0):
    # Haobo : This is to_coco situation #2
    # idx for stuff will be sem * div
    # Datasets: Cityscapes-DVPS
    pan_new = - np.ones_like(pan_map)

    thing_mapper = {CLASSES.index(itm): idx for idx, itm in enumerate(THING_CLASSES)}
    stuff_mapper = {CLASSES.index(itm): idx + NUM_THING for idx, itm in enumerate(STUFF_CLASSES)}
    mapper = {**thing_mapper, **stuff_mapper}
    for idx in np.unique(pan_map):
        if idx == NO_OBJ * DIVISOR_PAN:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = mapper[cls_id]
            inst_id = idx % DIVISOR_PAN
            if cls_id in stuff_mapper:
                assert inst_id == 0
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
    assert -1. not in np.unique(pan_new)
    return pan_new


@DATASETS.register_module()
class CityscapesDVPSDataset:
    CLASSES = build_classes()
    PALETTE = build_palette()

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 split='train',
                 ref_sample_mode: Literal['random', 'sequence', 'test'] = 'sequence',
                 ref_seq_index: List[int] = None,
                 ref_seq_len_test: int = 4,
                 with_depth: bool = False
                 ):
        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        video_seq_dir = os.path.join(data_root, 'video_sequence', split)
        assert os.path.exists(video_seq_dir)
        assert 'leftImg8bit' not in video_seq_dir

        # Dataset information
        # 8 + 11 for Cityscapes-DVPS; 255 for no_obj
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert self.num_classes == len(self.CLASSES)
        self.no_obj_class = NO_OBJ_HB

        # ref_seq_index is None means no ref img
        self.ref_sample_mode = ref_sample_mode
        if ref_seq_index is None:
            ref_seq_index = []
        self.ref_seq_index = ref_seq_index

        filenames = list(map(lambda x: str(x), os.listdir(video_seq_dir)))
        img_names = sorted(list(filter(lambda x: 'leftImg8bit' in x, filenames)))

        images = []
        for itm in img_names:
            seq_id, img_id, location, _, _, _ = itm.split(sep="_", maxsplit=5)
            item_full = os.path.join(video_seq_dir, itm)
            images.append(SeqObj({
                'seq_id': int(seq_id),
                'img_id': int(img_id),
                'location': location,
                'img': item_full,
                'depth': item_full.replace('leftImg8bit', 'depth') if with_depth else None,
                'ann': item_full.replace('leftImg8bit', 'gtFine_instanceTrainIds'),
                'no_obj_class': self.no_obj_class
            }))
            assert os.path.exists(images[-1]['img'])
            if not test_mode:
                if with_depth:
                    assert os.path.exists(images[-1]['depth'])
                assert os.path.exists(images[-1]['ann'])

        # Warning from Haobo: the following codes are dangerous
        # because they rely on a consistent seed among different
        # processes. Please contact me before using it.
        reference_images = {hash(image): image for image in images}

        sequences = []
        if self.ref_sample_mode == 'random':
            for img_cur in images:
                is_seq = True
                seq_now = [img_cur.dict]
                if self.ref_seq_index:
                    for index in random.choices(self.ref_seq_index, k=1):
                        query_obj = SeqObj({
                            'seq_id': img_cur.dict['seq_id'],
                            'img_id': img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(reference_images[hash(query_obj)].dict)
                        else:
                            is_seq = False
                            break
                if is_seq:
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'sequence':
            # In the sequence mode, the first frame is the key frame
            # Note that sequence mode may have multiple pointer to one frame
            for img_cur in images:
                is_seq = True
                seq_now = []
                if self.ref_seq_index:
                    for index in reversed(self.ref_seq_index):
                        query_obj = SeqObj({
                            'seq_id': img_cur.dict['seq_id'],
                            'img_id': img_cur.dict['img_id'] + index
                        })
                        if hash(query_obj) in reference_images:
                            seq_now.append(copy.deepcopy(reference_images[hash(query_obj)].dict))
                        else:
                            is_seq = False
                            break
                if is_seq:
                    seq_now.append(copy.deepcopy(img_cur.dict))
                    seq_now.reverse()
                    sequences.append(seq_now)
        elif self.ref_sample_mode == 'test':
            if ref_seq_len_test == 0:
                sequences = [[copy.deepcopy(itm.dict)] for itm in images]
            elif ref_seq_len_test == 1:
                sequences = [[copy.deepcopy(itm.dict), copy.deepcopy(itm.dict)] for itm in images]
            else:
                seq_id_pre = -1
                seq_now = []
                for img_cur in images:
                    seq_id_now = img_cur.dict['seq_id']
                    if seq_id_now != seq_id_pre:
                        seq_id_pre = seq_id_now
                        if len(seq_now) > 0:
                            while len(seq_now) < ref_seq_len_test + 1:
                                seq_now.append(copy.deepcopy(seq_now[-1]))
                            sequences.append(seq_now)
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    elif len(seq_now) % (ref_seq_len_test + 1) == 0:
                        sequences.append(seq_now)
                        seq_now = [copy.deepcopy(img_cur.dict), copy.deepcopy(img_cur.dict)]
                    else:
                        seq_now.append(copy.deepcopy(img_cur.dict))
        elif self.ref_sample_mode == 'img':
            sequences = [img_cur.dict for img_cur in images]
        else:
            raise ValueError("{} not supported.".format(self.ref_sample_mode))

        self.sequences = sequences
        self.images = reference_images

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

        # for all_val
        self.split = split
        self.logger = get_root_logger()

    def pre_pipelines(self, results):
        if isinstance(results, List):
            for _results in results:
                _results['img_info'] = []
                _results['thing_lower'] = 0
                _results['thing_upper'] = self.num_thing_classes
                _results['ori_filename'] = os.path.basename(_results['img'])
                _results['filename'] = _results['img']
                _results['pre_hook'] = to_coco
        else:
            assert self.test_mode
            results['img_info'] = []
            results['thing_lower'] = 0
            results['thing_upper'] = self.num_thing_classes
            results['ori_filename'] = os.path.basename(results['img'])
            results['filename'] = results['img']
            results['pre_hook'] = to_coco

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results)
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # Copy and Modify from mmdet
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def __len__(self):
        """Total number of samples of data."""
        return len(self.sequences)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    def pre_eval(self, result, save_dir, seq_id, img_id):
        assert self.ref_sample_mode == 'img'
        max_ins = INSTANCE_DIVISOR
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=True, mode='direct', divisor=max_ins, with_ps_id=True)
        ])
        sem_seg = result['sem'].astype(np.int64)
        ins_map = result['track'].astype(np.int64)
        dep_map = result['depth']
        pred_pan = sem_seg * max_ins + ins_map
        torch.save(
            {"panseg": pred_pan.astype(np.uint32), "depth": dep_map.astype(np.float32)},
            os.path.join(save_dir, "pred", "{:06d}_{:06d}.pth".format(seq_id, img_id)),
        )

        img_info = copy.deepcopy(self.images[hash(SeqObj({'seq_id': seq_id, 'img_id': img_id}))].dict)
        self.pre_pipelines(img_info)
        gt = pipeline(img_info)
        gt_pan = gt['gt_panoptic_seg']
        gt_depth = gt['gt_depth']
        torch.save(
            {"panseg": gt_pan.astype(np.uint32), "depth": gt_depth.astype(np.float32)},
            os.path.join(save_dir, "gt", "{:06d}_{:06d}.pth".format(seq_id, img_id)),
        )

    # The evaluate func
    def evaluate(
            self,
            results,
            **kwargs
    ):
        # only support image test now
        # self.ref_sample_mode == 'test' is for video test
        assert self.ref_sample_mode == 'img'
        max_ins = INSTANCE_DIVISOR
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect', with_depth=True, mode='direct', divisor=max_ins, with_ps_id=True)
        ])

        pq_preds = []
        dp_preds = []
        for itm in results:
            _, _, seg_results, _, depth = itm
            if seg_results is not None:
                inst_map, seg_info = seg_results
                cat_map = np.zeros_like(inst_map) + self.num_thing_classes + self.num_stuff_classes
                for instance in seg_info:
                    cat_cur = instance['category_id']
                    assert cat_cur < self.num_thing_classes + self.num_stuff_classes
                    cat_map[inst_map == instance['id']] = cat_cur
                    if not instance['isthing']:
                        inst_map[inst_map == instance['id']] = 0
                pq_preds.append(cat_map.astype(np.int32) * max_ins + inst_map.astype(np.int32))
            dp_preds.append(depth)

        pq_gts = []
        dp_gts = []
        for itm in self.sequences:
            img_info = copy.deepcopy(itm)
            self.pre_pipelines(img_info)
            gt = pipeline(img_info)
            ps_id = gt['gt_panoptic_seg'].astype(np.int64)
            depth_gt = gt['gt_depth']
            pq_gts.append(ps_id)
            dp_gts.append(depth_gt)

        vpq_results = defaultdict(lambda: [])
        depth_metrics = defaultdict(lambda: [])
        for pq_pred, pq_gt, dp_pred, dp_gt in zip(pq_preds, pq_gts, dp_preds, dp_gts):
            depth_metric = compute_errors(dp_pred, dp_gt)
            for metric in depth_metric:
                depth_metrics[metric].append(depth_metric[metric])
            for depth_thr in [0., 0.5, 0.25, 0.1]:
                pq_pred_cur = copy.deepcopy(pq_pred)
                if depth_thr > 0.:
                    depth_mask = dp_gt > 0.
                    masked_pq_pred = pq_pred_cur[depth_mask]
                    ignored_pred_mask = \
                        (np.abs(dp_pred[depth_mask] - dp_gt[depth_mask]) / dp_gt[depth_mask]) > depth_thr
                    masked_pq_pred[ignored_pred_mask] = self.num_classes * max_ins
                    pq_pred_cur[depth_mask] = masked_pq_pred
                _result = vpq_eval([pq_pred_cur, pq_gt],
                                   num_classes=self.num_classes, max_ins=max_ins, ign_id=NO_OBJ_HB)
                vpq_results[depth_thr].append(_result)

        depth_str = ''
        for metric in depth_metrics:
            value = np.stack(depth_metrics[metric]).mean(axis=0)
            depth_str += '{}: {:.4f}; '.format(metric, value)
        pq_all = None

        for depth_thr in [0., 0.5, 0.25, 0.1]:
            symbol = "inf" if depth_thr == 0 else "{:.4f}".format(depth_thr)
            self.logger.info("-------------Results for DVPQ (lambda : {})".format(symbol))
            results = vpq_results[depth_thr]
            iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)[:self.num_classes]
            tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)[:self.num_classes]
            fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)[:self.num_classes]
            fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)[:self.num_classes]
            epsilon = 0.
            sq = iou_per_class / (tp_per_class + epsilon)
            rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + epsilon)
            pq = sq * rq
            self.logger.info("PQ@{}: {}".format(symbol, pq.tolist()))
            pq = np.nan_to_num(pq)
            if depth_thr == 0:
                pq_all = pq.mean()
            self.logger.info("PQ_all@{}: {:.4f}; PQ_th@{}: {:.4f}; PQ_st@{}: {:.4f}".format(
                symbol,
                pq.mean(),
                symbol,
                pq[:self.num_thing_classes].mean(),
                symbol,
                pq[self.num_thing_classes:].mean()
            ))
        self.logger.info(depth_str)
        return {
            "PQ_all": pq_all,
        }
