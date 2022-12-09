FROM python:3.9.12

# install ujson tqdm
RUN python -m pip install ujson tqdm

# install pytorch
RUN python -m pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# mmcv : 1.4.4
# mmcv 1.4.4 fixes a crucial bug (scatter)
# https://github.com/open-mmlab/mmcv/pull/1621
RUN until python -m pip install opencv-python-headless && MMCV_WITH_OPS=1 pip install git+https://github.com/open-mmlab/mmcv.git@1231ef1cfe2b2ff1e186cb3f3622a5072091012e; do sleep 0.1; done

# timm : 2201
RUN until python -m pip install git+https://github.com/rwightman/pytorch-image-models.git@ccfeb06936549f19c453b7f1f27e8e632cfbe1c2; do sleep 0.1; done

# install panopticapi
RUN python -m pip install git+https://github.com/cocodataset/panopticapi.git

# intsall pycocotools
RUN python -m pip install pycocotools terminaltables scipy

WORKDIR /opt/project

CMD [ "/bin/bash" ]
