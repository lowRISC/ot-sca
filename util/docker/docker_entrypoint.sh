#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Entrypoint for OpenTitan SCA/FI image. This script:
# - Replaces the current process with a new shell.

readonly SHELL='/bin/bash'

# Switch to shell
exec "${SHELL}"
