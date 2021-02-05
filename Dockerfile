FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get isntall -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 1" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# basic tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        zsh \
        vim \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \


# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        notebook \
        && \

# ==================================================================
# onnx
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        protobuf-compiler \
        libprotoc-dev \
        && \

    $PIP_INSTALL \
        --no-binary onnx onnx \
        && \

    $PIP_INSTALL \
        onnxruntime \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \

    $PIP_INSTALL \
        torch==1.7.1+cu110 \
        torchvision==0.8.2+cu110 \
        torchaudio===0.7.2 \
        -f https://download.pytorch.org/whl/torch_stable.html \
        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

    $PIP_INSTALL \
        tensorflow-gpu==2.4.0 \
        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

    $PIP_INSTALL \
        h5py \
        keras \
        && \

# ==================================================================
# opencv
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        && \

    $GIT_CLONE --branch 4.5.1 https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          .. && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2 && \

# ==================================================================
# Frameworks / Libraries (ML)
# ------------------------------------------------------------------
# pytorch-lightning: High level pytorch wrapper
# seaborn: High level matplotlib wrapper
# detectron2: PyTorch object detection framework
# od: Tensorflow object detection API
# vissl: Self-Supervised learning framework for PyTorch
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/tensorflow/models.git ~/models && \
    cd ~/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    $PIP_INSTALL --use-feature=2020-resolver . && \

    $GIT_CLONE git clone https://github.com/NVIDIA/apex ~/apex && \
    cd ~/apex && \
    $PIP_INSTALL --disable-pip-version-check --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \

    $PIP_INSTALL \
        pytorch-lightning \
        seaborn \
        vissl \
        'git+https://github.com/facebookresearch/detectron2.git' \
        && \


# ==================================================================
# Additional Tools (ML)
# ------------------------------------------------------------------
# cityscapesScripts: Cityscapes dataset utils
# pycocotools: COCO utils
# gpustat: gpu monitor tool
# ------------------------------------------------------------------

    $PIP_INSTALL \
        cityscapesscripts \
        pycocotools \
        gpustat \
        && \


# ==================================================================
# Additional Tools (Misc)
# ------------------------------------------------------------------
# Tools I prefer personally, nothing is mandatory
# ------------------------------------------------------------------
# tmux: Terminal multiplexer
# bat: Inhanced cat
# fzf: Commandline fuzzy finder
# exa: Inhanced ls
# broot: Inhanced tree
# ripgrep: Inhanced grep
# ------------------------------------------------------------------

    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" && \


    $APT_INSTALL \
        bat \
        tmux \
        exa \
        ripgrep \
        && \

    wget https://dystroy.org/broot/download/x86_64-linux/broot /usr/local/bin/broot && \
    chmod +x /usr/local/bin/broot && \ 

    $GIT_CLONE https://github.com/junegunn/fzf.git ~/.fzf && \
    yes | ~/.fzf/install && \

    $GIT_CLONE https://github.com/zsh-users/zsh-autosuggestions $ZSH/plugins/zsh-autosuggestions && \
    $GIT_CLONE https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \

    wget https://github.com/ryanking13/dotfiles/blob/master/.vimrc ~/.vimrc && \
    wget https://github.com/ryanking13/dotfiles/blob/master/.zshrc ~/.zshrc && \

    vim -c 'qa!' && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 6006