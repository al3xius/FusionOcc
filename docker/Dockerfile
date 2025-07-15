FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8


RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

### install mmcv-full
RUN pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu11.3/torch1.10.0/index.html
RUN pip install mmdet==2.25.1 mmsegmentation==0.25.0

RUN pip install torch-scatter==2.0.9

WORKDIR /roadrunner
COPY ../ /roadrunner

RUN pip install -r /roadrunner/requirements.txt
