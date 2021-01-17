#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Entrypoint for OpenTitan SCA/FI image. This script:
# - Creates a new non-privileged user with the same UID and GID as the owner of
#   the host directory to avoid permission issues,
# - Adds this user to plugdev and dialout groups to able to access
#   chipwhisperer devices,
# - Drops root privileges by switching to the newly created user, and
# - Replaces the current process with a new shell.

# We expect only two variables.
if [[ "$#" -ne 2 ]]; then
  echo "Unexpected number of parameters: $#" >&2
  exit 1
fi

readonly USER_NAME="$1"
readonly MOUNT_DIR="$2"
readonly SHELL='/bin/bash'

# Create a user with the same UID and GID as the owner of the mount.
# Note: The user is also added to plugdev and dialout to be able talk to
# chipwhisperer USB devices. IDs of these groups must match those of the
# host system, which typically is the case.
HOST_UID="$(stat -c '%u' "${MOUNT_DIR}")"
readonly HOST_UID
HOST_GID="$(stat -c '%g' "${MOUNT_DIR}")"
readonly HOST_GID 
echo "Creating user '${USER_NAME}' with UID=${HOST_UID}, GID=${HOST_GID}."
groupadd -g "${HOST_GID}" "${USER_NAME}"
useradd -u "${HOST_UID}" -g "${HOST_GID}" -m -s "${SHELL}" "${USER_NAME}"

# Install git lfs
runuser "${USER_NAME}" -c 'git lfs install' > /dev/null

# Cleanup, drop privileges, and replace the current process with a new shell.
rm /docker_entrypoint.sh /docker_entrypoint_wrapper.sh
HOME_DIR="$(getent passwd "${USER_NAME}" | cut -d : -f 6)"
readonly HOME_DIR
HOME="${HOME_DIR}" exec setpriv --reuid="${HOST_UID}" --regid="${HOST_GID}" --inh-caps -all --init-group "${SHELL}"
