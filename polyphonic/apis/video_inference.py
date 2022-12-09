# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.utils import get_root_logger


def single_gpu_test(
        model,
        data_loader,
        show=False,
        out_dir=None,
        show_score_thr=0.3,
        eval_dir=None
):
    assert eval_dir is not None
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        assert len(data['img_metas']) == 1
        seq_id = data.pop('seq_id', None)[0].item()
        img_id = data.pop('img_id', None)[0].item()
        if img_id == 0:
            model.module.init_tracker()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        assert len(result) == 1, "Only support bs=1 for inference"

        data_loader.dataset.pre_eval(result[0], eval_dir, seq_id, img_id)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    results = None
    return results
