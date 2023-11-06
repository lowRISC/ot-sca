#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Simple script to test AES captures
# Create results folder.
mkdir -p figures_tmp

# AES
MODE="aes"
BOARD=cw310
declare -A aes_test_list
aes_test_list["aes-random"]=100
aes_test_list["aes-random-batch"]=1000
aes_test_list["aes-fvsr-key"]=100
aes_test_list["aes-fvsr-key-batch"]=1000

ARGS="--force-program-bitstream"
for test in ${!aes_test_list[@]}; do
  echo Testing ${test} on CW310 - `date`
  NUM_TRACES=${aes_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_aes_cw310.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv ./ci_projects/sample_traces_${MODE}.html figures_tmp/${test}_traces.html
  ARGS=""
done

# KMAC
MODE="kmac"
declare -A kmac_test_list
kmac_test_list["kmac-random"]=100
kmac_test_list["kmac-fvsr-key"]=100
kmac_test_list["kmac-fvsr-key-batch"]=1000

ARGS="--force-program-bitstream"
for test in ${!kmac_test_list[@]}; do
  echo Testing ${test} on ${BOARD} - `date`
  NUM_TRACES=${kmac_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_kmac_cw310.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_${MODE}.html figures_tmp/${test}_traces.html
  ARGS=""
done

# SHA3
MODE="sha3"
declare -A sha3_test_list
sha3_test_list["sha3-fvsr-data"]=100
sha3_test_list["sha3-fvsr-data-batch"]=1000

ARGS="--force-program-bitstream"
for test in ${!sha3_test_list[@]}; do
  echo Testing ${test} on ${BOARD} - `date`
  NUM_TRACES=${sha3_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_sha3_cw310.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_${MODE}.html figures_tmp/${test}_traces.html
  ARGS=""
done

# OTBN keygen
MODE="otbn_vertical_keygen"
declare -A otbn_keygen_test_list
otbn_keygen_test_list["otbn-vertical"]=100
otbn_keygen_test_list["otbn-vertical-batch"]=1000

ARGS="--force-program-bitstream"
for test in ${!otbn_keygen_test_list[@]}; do
  echo Testing ${MODE} ${test} on ${BOARD} - `date`
  NUM_TRACES=${otbn_keygen_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_otbn_vertical_keygen.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_ecdsa_keygen.html figures_tmp/${BOARD}_${MODE}_${test}_traces.html
  ARGS=""
done

# OTBN modinv
MODE="otbn_vertical_modinv"
declare -A otbn_modinv_test_list
otbn_modinv_test_list["otbn-vertical"]=100
for test in ${!otbn_modinv_test_list[@]}; do
  echo Testing ${MODE} ${test} on ${BOARD} - `date`
  NUM_TRACES=${otbn_modinv_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_otbn_vertical_modinv.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_ecdsa_modinv.html figures_tmp/${BOARD}_${MODE}_${test}_traces.html
done

# OTBN Horizonal ECDSA256
MODE="ecdsa256"
declare -A otbn_ecdsa256_test_list
otbn_ecdsa256_test_list["ecdsa-simple"]=10
otbn_ecdsa256_test_list["ecdsa-stream"]=10

ARGS="--force-program-bitstream"
for test in ${!otbn_ecdsa256_test_list[@]}; do
  echo Testing ${MODE} ${test} on ${BOARD} - `date`
  NUM_TRACES=${otbn_ecdsa256_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_ecdsa256_cw310.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_ecdsa256.html figures_tmp/${BOARD}_${MODE}_${test}_traces.html
  ARGS=""
done

# OTBN Horizonal ECDSA384
MODE="ecdsa384"
declare -A otbn_ecdsa384_test_list
otbn_ecdsa384_test_list["ecdsa-simple"]=10
otbn_ecdsa384_test_list["ecdsa-stream"]=10

for test in ${!otbn_ecdsa384_test_list[@]}; do
  echo Testing ${MODE} ${test} on ${BOARD} - `date`
  NUM_TRACES=${otbn_ecdsa384_test_list[${test}]}
  ../cw/capture.py --cfg-file ./cfg/ci_capture_ecdsa384_cw310.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS}
  mv projects/sample_traces_ecdsa384.html figures_tmp/${BOARD}_${MODE}_${test}_traces.html
done
