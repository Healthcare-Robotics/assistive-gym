FROM ubuntu:18.04

RUN apt-get update && apt-get -y install sudo
RUN apt-get -y upgrade && apt-get update && apt-get clean && apt-get -y install curl git build-essential zlib1g-dev libssl-dev libopenmpi-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx vim htop

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu && echo "ubuntu:ubuntu" | chpasswd
USER ubuntu
WORKDIR /home/ubuntu

RUN curl https://pyenv.run | bash
RUN echo '\n export PATH="~/.pyenv/bin:$PATH"\n eval "$(pyenv init -)"\n eval "$(pyenv virtualenv-init -)"' >> /home/ubuntu/.bashrc
ENV HOME  /home/ubuntu
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install 3.6.5 && pyenv global 3.6.5 && pip3 install pip --upgrade
RUN pip3 install screeninfo
# RUN pip3 install git+https://github.com/Zackory/bullet3.git
RUN git clone https://github.com/Healthcare-Robotics/assistive-gym.git && cd assistive-gym && pip3 install -e .
RUN pip3 install git+https://github.com/Zackory/pytorch-a2c-ppo-acktr --no-cache-dir
RUN pip3 install git+https://github.com/openai/baselines.git

# docker build -t "assistive-gym-v1.0:Dockerfile" .
# docker run -it 0f55f5d433e6 bash
# To reconnect using Xserver
# See: https://medium.com/@mreichelt/how-to-show-x11-windows-within-docker-on-mac-50759f4b65cb
# docker exec -it -e DISPLAY=host.docker.internal:0 ebccc26459f8 bash
# docker ps -a
# docker images
# docker commit ebeb779ed44e assistive-gym-v1.0:compiled
# docker save assistive-gym-v1.0:compiled | gzip > assistive-gym-v1_docker.tar.gz


# Installation

# sudo apt-get update
# sudo apt-get upgrade
# sudo apt-get install curl git build-essential zlib1g-dev libssl-dev libopenmpi-dev libglib2.0-0 libsm6 libxext6 libxrender-dev vim
# curl https://pyenv.run | bash
# vim ~/.bashrc
# # Add to ~/.bashrc
# export PATH="~/.pyenv/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

# source ~/.bashrc
# pyenv install 3.6.5
# pyenv local 3.6.5
# mkdir -p git/assistive-gym-1.0
# cd git/assistive-gym-1.0

# NOTE: Copy over custom files
# # docker cp assistive-gym ebccc26459f8:home/ubuntu/git/assistive-gym-1.0

# pip3 install screeninfo
# pip3 install git+https://github.com/Zackory/bullet3.git
# git clone https://github.com/Healthcare-Robotics/assistive-gym.git
# cd assistive-gym
# pip3 install -e .
# pip3 install git+https://github.com/Zackory/pytorch-a2c-ppo-acktr --no-cache-dir
# pip3 install git+https://github.com/openai/baselines.git

