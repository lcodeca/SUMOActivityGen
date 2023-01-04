# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:22.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Setup Environment and Install SUMO 
RUN apt update && apt --yes install software-properties-common && apt update
# these are the same SUMO dependencies for the Github Action
RUN apt --yes install cmake libeigen3-dev libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgtest-dev libgoogle-perftools-dev libgl2ps-dev python3-dev python3-setuptools swig openjdk-8-jdk maven ccache
# actual sumo installation
RUN add-apt-repository ppa:sumo/stable && apt update
RUN apt --yes install sumo sumo-tools

# Install Python3 pip
RUN apt --yes install python3-pip

# Install apt requirements for SAGA
RUN apt --yes install python3-rtree libspatialindex-dev

# Install pip requirements for SAGA
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install pip requirements for Devel & Testing
COPY requirements_dev.txt .
RUN python3 -m pip install -r requirements_dev.txt

WORKDIR /repo

# Creates a non-root user with an explicit UID and adds permission to access the /repo folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" developer && chown -R developer /repo
USER developer

CMD ["bash"]