# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path


def ap_check_file_exists(file_path: str) -> Path:
    """Verifies that the provided file path is valid

    Args:
        file_path: The file path.

    Returns:
        The file path.
    """
    path = Path(file_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File {path} does not exist")
    return path


def ap_check_dir_exists(path: str) -> Path:
    """Verifies that the provided path is valid

    Args:
        path: The path.

    Returns:
        The file path.
    """
    path = Path(path)
    if not path.parent.exists():
        print(f"Path {path.parent} does not exist, creating directory...")
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_arguments(argv):
    """ Command line argument parsing.

    Args:
        argv: The command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse")
    parser.add_argument("-c",
                        "--capture_config",
                        dest="cfg",
                        type=ap_check_file_exists,
                        required=True,
                        help="Path of the attack config file")
    parser.add_argument("-p",
                        "--project",
                        dest="project",
                        type=ap_check_dir_exists,
                        required=True,
                        help="Path of the output project directory")

    args = parser.parse_args(argv)

    return args
