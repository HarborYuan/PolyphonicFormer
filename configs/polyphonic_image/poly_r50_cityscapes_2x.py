_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/polyphonic_former.py',
    '../_base_/datasets/cityscapes_dvps.py',
]

load_from = 'https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/knet_r50_pt.pth'

data = dict(
    samples_per_gpu=1,
)
optimizer = dict(
    lr=0.0001,
)
