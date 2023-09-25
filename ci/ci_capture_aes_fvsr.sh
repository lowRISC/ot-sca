#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Simple script to test AES fvsr-key capture
../cw/capture.py --cfg-file ci_capture_aes_cw310.yaml capture aes-fvsr-key-batch
if [ -d "./ci_projects/opentitan_simple_aes_data" ];
then
    echo "Directory ./ci_projects/opentitan_simple_aes_data exists"
else
    echo "Error: Directory ./ci_projects/opentitan_simple_aes_data does not exists."
    exit 1
fi
