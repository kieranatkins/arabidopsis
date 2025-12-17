FROM python:3.11

# Device can be either CPU or CUDA. Default is to install CPU
# Currently, only non-cpu torch version support is CUDA 12.1.
ARG device=cpu
# ARG device=cu121

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 ninja-build -y
RUN python -m pip install torch==2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/$device
RUN python -m pip install openmim
RUN python -m mim install mmengine
# RUN python -m mim install "mmcv==2.1"
RUN pip install mmcv==2.1 -f https://download.openmmlab.com/mmcv/dist/$device/torch2.1/index.html
RUN git clone https://github.com/open-mmlab/mmdetection.git
WORKDIR /mmdetection
RUN python -m pip install -v -e .
RUN python -c "import mmdet;print(mmdet.__version__)"
RUN python mmdet/utils/collect_env.py
RUN python -m pip install "numpy<2" pandas pycocotools opencv-python scipy scikit-image sknw networkx alphashape


WORKDIR /app
COPY . .

ENTRYPOINT ["./run.sh"]