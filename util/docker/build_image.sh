#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

readonly IMAGE_NAME='ot-sca'
readonly DOCKERFILE='util/docker/Dockerfile'

docker build -t ${IMAGE_NAME} -f ${DOCKERFILE} .
