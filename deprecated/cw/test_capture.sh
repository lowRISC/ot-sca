#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Simple script to test all relevant capture modes supported by capture.py.
# To test capture modes on the CW310 board, type
#
# ./test_capture.sh
#

set -e
set -o pipefail

error () {
    echo >&2 "$@"
    exit 1
}

# Input arguments.
# Only one argument is supported to specify the FPGA board.
if [ $# -lt 1 ]; then
  BOARD=cw310
else
  BOARD=$1
fi
if `[ "${BOARD}" != "cw305" ] && [ "${BOARD}" != "cw310" ]`; then
  error "Board ${BOARD} not supported"
fi

# Create results folder.
mkdir -p tmp

# AES
MODE="aes"
declare -A aes_test_list
aes_test_list["aes-random"]=100
aes_test_list["aes-random-batch"]=1000
aes_test_list["aes-fvsr-key"]=100
aes_test_list["aes-fvsr-key-batch"]=1000

ARGS="--force-program-bitstream"
for test in ${!aes_test_list[@]}; do
  echo Testing ${test} on ${BOARD}
  echo Testing ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
  NUM_TRACES=${aes_test_list[${test}]}
  ./capture.py --cfg-file capture_${MODE}_${BOARD}.yaml capture ${test} \
      --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
  mv projects/sample_traces_${MODE}.html tmp/${BOARD}_${test}_traces.html
  ARGS=""
done

if [ ${BOARD} == "cw310" ]; then
  # KMAC
  MODE="kmac"
  declare -A kmac_test_list
  kmac_test_list["kmac-random"]=100
  kmac_test_list["kmac-fvsr-key"]=100
  kmac_test_list["kmac-fvsr-key-batch"]=1000

  ARGS="--force-program-bitstream"
  for test in ${!kmac_test_list[@]}; do
    echo Testing ${test} on ${BOARD}
    echo Testing ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${kmac_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}_${BOARD}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_${MODE}.html tmp/${BOARD}_${test}_traces.html
    ARGS=""
  done

  # SHA3
  MODE="sha3"
  declare -A sha3_test_list
  sha3_test_list["sha3-fvsr-data"]=100
  sha3_test_list["sha3-fvsr-data-batch"]=1000

  ARGS="--force-program-bitstream"
  for test in ${!sha3_test_list[@]}; do
    echo Testing ${test} on ${BOARD}
    echo Testing ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${sha3_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}_${BOARD}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_${MODE}.html tmp/${BOARD}_${test}_traces.html
    ARGS=""
  done

  # OTBN keygen
  MODE="otbn_vertical_keygen"
  declare -A otbn_keygen_test_list
  otbn_keygen_test_list["otbn-vertical"]=100
  otbn_keygen_test_list["otbn-vertical-batch"]=1000

  ARGS="--force-program-bitstream"
  for test in ${!otbn_keygen_test_list[@]}; do
    echo Testing ${MODE} ${test} on ${BOARD}
    echo Testing ${MODE} ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${otbn_keygen_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_ecdsa_keygen.html tmp/${BOARD}_${MODE}_${test}_traces.html
    ARGS=""
  done

  # OTBN modinv
  MODE="otbn_vertical_modinv"
  declare -A otbn_modinv_test_list
  otbn_modinv_test_list["otbn-vertical"]=100
  for test in ${!otbn_modinv_test_list[@]}; do
    echo Testing ${MODE} ${test} on ${BOARD}
    echo Testing ${MODE} ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${otbn_modinv_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_ecdsa_modinv.html tmp/${BOARD}_${MODE}_${test}_traces.html
  done

  # OTBN horizontal tests

  # ECDSA256
  MODE="ecdsa256"
  declare -A otbn_ecdsa256_test_list
  otbn_ecdsa256_test_list["ecdsa-simple"]=10
  otbn_ecdsa256_test_list["ecdsa-stream"]=10

  ARGS="--force-program-bitstream"
  for test in ${!otbn_ecdsa256_test_list[@]}; do
    echo Testing ${MODE} ${test} on ${BOARD}
    echo Testing ${MODE} ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${otbn_ecdsa256_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}_${BOARD}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_ecdsa256.html tmp/${BOARD}_${MODE}_${test}_traces.html
    ARGS=""
  done

  # ECDSA384
  MODE="ecdsa384"
  declare -A otbn_ecdsa384_test_list
  otbn_ecdsa384_test_list["ecdsa-simple"]=10
  otbn_ecdsa384_test_list["ecdsa-stream"]=10

  for test in ${!otbn_ecdsa384_test_list[@]}; do
    echo Testing ${MODE} ${test} on ${BOARD}
    echo Testing ${MODE} ${test} on ${BOARD} - `date` >> "tmp/${BOARD}_test_capture.log"
    NUM_TRACES=${otbn_ecdsa384_test_list[${test}]}
    ./capture.py --cfg-file capture_${MODE}_${BOARD}.yaml capture ${test} \
        --num-traces ${NUM_TRACES} ${ARGS} &>> "tmp/${BOARD}_test_capture.log"
    mv projects/sample_traces_ecdsa384.html tmp/${BOARD}_${MODE}_${test}_traces.html
  done
fi

echo "Done! Checkout tmp/${BOARD}_... for results."
