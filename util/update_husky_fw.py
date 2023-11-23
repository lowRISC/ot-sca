#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse

import chipwhisperer as cw


def parse_arguments(argv):
    """ Command line argument parsing.

    Args:
        argv: The command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse")
    parser.add_argument("-s",
                        "--sn",
                        dest="sn",
                        type=int,
                        required=False,
                        help="Serial number of CW Husky")

    args = parser.parse_args(argv)

    return args


def update_fw(argv=None) -> None:
    """ Check ChipWhisperer API version.

    Read CW API version and compare against expected version.

    Args:
        cw_version_exp: Expected CW version.

    Returns:
        Raises a runtime error on a mismatch.
    """
    # Parse the provided arguments.
    args = parse_arguments(argv)
    # Init scope with SN, if provided.
    if args.sn:
        scope = cw.scope(sn=str(args.sn))
    else:
        scope = cw.scope()

    # Upgrade firmware.
    scope.upgrade_firmware()


if __name__ == "__main__":
    update_fw()
