# Using cuda base image. It works with M1 too, but no GPU.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -y install sudo
RUN apt-get -y upgrade && apt-get update && apt-get clean && apt-get -y install curl git build-essential zlib1g-dev libssl-dev libopenmpi-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx vim htop
RUN apt-get -y install python3-dev python3 python3-venv python3-pip
RUN apt-get -y install pkg-config libhdf5-dev
RUN apt-get -y install python3-h5py

SHELL ["bash", "-l" ,"-c"]
WORKDIR /opt/assistive-gym

RUN pip3 install -U setuptools pip

# Required for aarch64
RUN pip3 install 'Cython<3'

ENV 

RUN pip3 install screeninfo
RUN mkdir -p /opt/assistive-gym
COPY . /opt/assistive-gym
RUN cd /opt/assistive-gym && pip3 install -e . && pip3 install -U h5py

# To reconnect using Xserver
# See: https://medium.com/@mreichelt/how-to-show-x11-windows-within-docker-on-mac-50759f4b65cb
# docker exec -it -e DISPLAY=host.docker.internal:0 ebccc26459f8 bash
# docker ps -a
# docker images
# docker commit ebeb779ed44e assistive-gym-v1.0:compiled
# docker save assistive-gym-v1.0:compiled | gzip > assistive-gym-v1_docker.tar.gz
