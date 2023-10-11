#!/bin/bash

# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Simple script to test AES capture.
mkdir -p tmp

# AES
MODE="aes"
BOARD=cw310
declare -A aes_test_list
aes_test_list["aes-static"]=100

ARGS="--force-program-bitstream"
for test in ${!aes_test_list[@]}; do
  echo Testing ${test} on CW310 - `date`
  NUM_TRACES=${aes_test_list[${test}]}
  ../cw/capture.py --cfg-file ci_capture_aes_cw310.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/test_capture.log"

  mv ./ci_projects/sample_traces_${MODE}.html tmp/${test}_traces.html
  ARGS=""
done
