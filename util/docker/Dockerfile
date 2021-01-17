# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Docker file for OpenTitan side channel analysis and fault injection.

FROM ubuntu:18.04

LABEL version="0.1"
LABEL description="OpenTitan SCA/FI image"
LABEL maintainer="alphan@google.com"

ARG USER_NAME=ot
ARG MOUNT_DIR=/repo
ARG TIME_ZONE=America/New_York 
# This is convenient if we want to switch to different python version.
# Note: numba, a chipwhisperer dependency, requires python 3.7+ but does not
# support python 3.9 yet.
ARG PYTHON=python3.8
ARG VENV_PATH=/opt/venv

# Use bash as the default shell.
SHELL ["/bin/bash", "-c"]

# Imstall required packages.
# git-lfs: https://github.com/git-lfs/git-lfs/wiki/Installation#docker-recipes
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        git \
        curl \
        ca-certificates \
        screen \
        locales \
        tzdata \
        setpriv && \
    curl -fsSL https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | DEBIAN_FRONTEND="noninteractive" bash && \
    apt-get update && \
    apt-get install --no-install-recommends -y git-lfs

# Set locale and time zone.
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
RUN ln -fs /usr/share/zoneinfo/"${TIME_ZONE}" /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

# Chipwhisperer dependencies.
# https://chipwhisperer.readthedocs.io/en/latest/prerequisites.html
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    libusb-dev \
    make

# Python virtual environment and dependencies.
# Note: Permissions are relaxed so that the user created in the entrypoint can also use `pip`.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    "${PYTHON}" \
    "${PYTHON}"-dev \
    "${PYTHON}"-distutils \
    "${PYTHON}"-venv
RUN "${PYTHON}" -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}"/bin:"${PATH}"
ENV VIRTUAL_ENV="${VENV_PATH}"
COPY python-requirements.txt /tmp/python-requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/python-requirements.txt && \
    chmod -R o=u "${VENV_PATH}";

# Cleanup
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/*

# Entrypoint
COPY util/docker/docker_entrypoint.sh /docker_entrypoint.sh
RUN echo "exec /docker_entrypoint.sh '${USER_NAME}' '${MOUNT_DIR}'" > /docker_entrypoint_wrapper.sh
RUN chmod +x /docker_entrypoint.sh /docker_entrypoint_wrapper.sh
ENTRYPOINT /docker_entrypoint_wrapper.sh
