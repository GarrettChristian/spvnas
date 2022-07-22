


# Base Image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Install Python
RUN apt update && apt install python3-pip -y

RUN pip3 install --upgrade pip

# Model Dependencies
# RUN pip3 install torch==1.9.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu100
# RUN pip3 install torchvision==0.10.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu100
RUN pip3 install pillow==6.1.0
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchaudio==0.9.1
RUN pip3 install numba==0.53.1
RUN pip3 install opencv-python==4.5.5.64
RUN pip3 install torchpack==0.3.1
RUN pip3 install tqdm==4.64.0

# MPI (for torch sparse)
RUN apt-get install -y ninja-build
RUN apt-get install -y libboost-all-dev
RUN pip3 install mpi4py==3.1.3


# Install custom torch sparse 
RUN apt update && apt install libsparsehash-dev
RUN apt-get install -y git

# ENV rather than export
# https://stackoverflow.com/questions/27093612/in-a-dockerfile-how-to-update-path-environment-variable
ENV PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CUDA_HOME=/usr/local/cuda-11.1
ENV CUDA_PATH=/usr/local/cuda-11.1


# RUN pip3 install tensorflow==2.6.2
RUN pip3 install tensorflow==2.4.0

# RUN export CUDA_HOME=/usr/local/cuda

# commit 4fa67d2f728fa78b1748ba065b0e6ff1ae528eea
# https://github.com/mit-han-lab/torchsparse/blob/master/docs/FAQ.md
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# NOTE MY GPU WAS A GTX 1080 aka SM_61 for TORCH_CUDA_ARCH_LIST
# RUN pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN TORCH_CUDA_ARCH_LIST="6.1" FORCE_CUDA=1 pip3 install -v git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0


RUN nvcc --version

RUN echo $PATH

RUN ls /usr/local




# # Base Image
# FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# # Install Python
# RUN apt update && apt install python3-pip -y

# RUN pip3 install --upgrade pip

# # Model Dependencies
# # RUN pip3 install torch==1.9.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu100
# # RUN pip3 install torchvision==0.10.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu100
# RUN pip3 install pillow==6.1.0
# RUN pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install torchaudio==0.9.1
# RUN pip3 install numba==0.53.1
# RUN pip3 install opencv-python==4.5.5.64
# RUN pip3 install torchpack==0.3.1
# RUN pip3 install tqdm==4.64.0

# # MPI (for torch sparse)
# RUN apt-get install -y ninja-build
# RUN apt-get install -y libboost-all-dev
# RUN pip3 install mpi4py==3.1.3


# # Install custom torch sparse 
# RUN apt update && apt install libsparsehash-dev
# RUN apt-get install -y git

# RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# RUN export CUDA_HOME=/usr/local/cuda

# # RUN pip3 install tensorflow==2.6.2
# RUN pip3 install tensorflow==2.3.0

# # RUN export CUDA_HOME=/usr/local/cuda

# # commit 4fa67d2f728fa78b1748ba065b0e6ff1ae528eea
# # RUN pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
# RUN pip3 install -v git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0


# RUN nvcc --version

# RUN echo $PATH



# RUN ls /usr/local/cuda