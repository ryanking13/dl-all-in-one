FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir --use-deprecated=legacy-resolver install --upgrade " && \
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
        cmake \
        && \

#    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
#    cd ~/cmake && \
#    ./bootstrap && \
#    make -j"$(nproc)" install && \

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
        tmux \
        && \

    wget https://github.com/sharkdp/bat/releases/download/v0.18.0/bat_0.18.0_amd64.deb && \
    dpkg -i bat_0.18.0_amd64.deb && \
    rm -f bat_0.18.0_amd64.deb && \
 
    wget https://dystroy.org/broot/download/x86_64-linux/broot -O /usr/local/bin/broot && \
    chmod +x /usr/local/bin/broot && \ 

    wget https://github.com/ogham/exa/releases/download/v0.9.0/exa-linux-x86_64-0.9.0.zip && \
    unzip exa-linux-x86_64-0.9.0.zip && \
    rm -f exa-linux-x86_64-0.9.0.zip && \
    mv exa-linux-x86_64 /usr/local/bin/exa && \
    chmod +x /usr/local/bin/exa && \

    wget https://github.com/BurntSushi/ripgrep/releases/download/12.1.1/ripgrep_12.1.1_amd64.deb && \
    dpkg -i ripgrep_12.1.1_amd64.deb && \
    rm -f ripgrep_12.1.1_amd64.deb && \

    $GIT_CLONE https://github.com/junegunn/fzf.git ~/.fzf && \
    yes | ~/.fzf/install && \

    $GIT_CLONE https://github.com/zsh-users/zsh-autosuggestions $ZSH/plugins/zsh-autosuggestions && \
    $GIT_CLONE https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \

    wget https://github.com/ryanking13/dotfiles/blob/master/.vimrc ~/.vimrc && \
    wget https://github.com/ryanking13/dotfiles/blob/master/.zshrc ~/.zshrc && \

    vim -c 'qa!' && \

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
        opencv-contrib-python \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        notebook \
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
        tensorflow==2.4.1 \
        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

    $PIP_INSTALL \
#        h5py \
        keras \
        && \

# ==================================================================
# Frameworks / Libraries (ML)
# ------------------------------------------------------------------
# pytorch-lightning: High level pytorch wrapper
# seaborn: High level matplotlib wrapper
# detectron2: PyTorch object detection framework
# od: Tensorflow object detection API
# vissl: Self-Supervised learning framework for PyTorch
# apex:
# horovod: 
# timm:
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        protobuf-compiler \
        libprotoc-dev \
        ninja-build \
        && \


    $GIT_CLONE https://github.com/tensorflow/models.git ~/models && \
    cd ~/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    $PIP_INSTALL . && \

    $GIT_CLONE https://github.com/NVIDIA/apex ~/apex && \
    cd ~/apex && \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0" $PIP_INSTALL --disable-pip-version-check --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \

    HOROVOD_GPU_OPERATIONS=NCCL $PIP_INSTALL \
        pytorch-lightning \
        seaborn \
        # vissl \
        horovod \
        timm \
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
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 6006
