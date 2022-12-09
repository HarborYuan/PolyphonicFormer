FROM nvcr.io/nvidia/pytorch:21.10-py3

# install ujson
RUN pip install ujson

# handle the timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="PRC" apt-get install tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && unlink /etc/localtime && ln -s /usr/share/zoneinfo/PRC /etc/localtime

# mmcv : 1.3.18
RUN until MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@58e32423f04048980946693772fb059d554eacd0; do sleep 0.1; done

# mmdet : 2.20.0
# This pakcage has already incorporated into project.
# RUN until FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmdetection.git@ff9bc39913cb3ff5dde79d3933add7dc2561bab7; do sleep 0.1; done

# panopticapi
RUN until python -m pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0; do sleep 0.1; done

# timm : 2201
RUN until python -m pip install git+https://github.com/rwightman/pytorch-image-models.git@ccfeb06936549f19c453b7f1f27e8e632cfbe1c2; do sleep 0.1; done

# mmseg : 0.20.0
# This pakcage has already incorporated into project.
# RUN until python -m pip install git+https://github.com/open-mmlab/mmsegmentation.git@bdfbaf00c1f55e054e6088e60f955345ab7383c0; do sleep 0.1; done

# install misc packages
RUN python -m pip install terminaltables

WORKDIR /opt/project

