#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

readonly CONTAINER_WORK_DIR=/repo
readonly CONTAINER_NAME=ot-sca

function usage() {
  cat <<USAGE

Run OpenTitan SCA/FI image.

Usage: $0 -d DEVICE [-d DEVICE] -m SHM_SIZE -w HOST_WORK_DIR [-h]
 
  -d: Host device to be added to the container. This option can be used multiple times.
  -m: Shared memory size (/dev/shm) of the container. Should be at least 1/3 of total memory.
  -w: Host directory that will be mounted into the container as /repo.
  -h: Print usage information and exit.

USAGE
}

function error() {
  echo "$1" 2>&1
  usage
  exit 1
}

DEVICES=()
while getopts ':d:m:w:h' opt; do
  case "${opt}" in
    d)  DEVICES+=("${OPTARG}") ;;
    m)  SHM_SIZE="${OPTARG}" ;;
    w)  HOST_WORK_DIR="${OPTARG}" ;;
    h)  usage; exit 0 ;;
    :)  error "Option '-${OPTARG}' requires an argument." ;;
    \?) error "Invalid option: '-${OPTARG}'" ;;
    *)  error "Invalid option: '-${opt}'" ;;
  esac
done
readonly DEVICES
readonly SHM_SIZE
readonly HOST_WORK_DIR

# Make sure that there are no additional arguments.
shift $((OPTIND-1))
if [[ "$#" -gt 0 ]]; then
  error "Unexpected arguments: '$*'"
fi

# Make sure that all required options are present.
if [[ -z "${HOST_WORK_DIR}" ]] || [[ -z "${SHM_SIZE}" ]] || [[ ${#DEVICES[@]} -eq 0 ]]; then
  error "Missing options: '-m SHM_SIZE', '-w HOST_WORK_DIR', and '-d DEVICE' are required."
fi

docker run --rm -it \
    --shm-size "${SHM_SIZE}" \
    -v "${HOST_WORK_DIR}":"${CONTAINER_WORK_DIR}" \
    -w "${CONTAINER_WORK_DIR}" \
    "${DEVICES[@]/#/--device=}" \
    --hostname "${CONTAINER_NAME}" --name "${CONTAINER_NAME}" "${CONTAINER_NAME}"
