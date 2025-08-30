# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import zlib
from pathlib import Path

import git


def get_git_hash() -> str:
    """Get the Git hash of the repository

    Returns:
        String containing the Git hash.
    """
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def file_crc(file_path: Path) -> int:
    """Calculates CRC32 of the provided file

    Args:
        file_path: The path of the file to calculate the CRC32.

    Returns:
        Int containing the CRC32 of the file.
    """
    crc_str = 0
    with open(file_path, "rb") as f:
        # Read 1MiB of the binary & calculate CRC32.
        while chunk := f.read(1024 * 1024):
            crc_str = zlib.crc32(chunk, crc_str)
    return crc_str


def get_binary_blob(file_path: Path) -> bytearray:
    """Read binary blob from the provided file

    Args:
        file_path: The path of the file,

    Returns:
        Bytearray containing the bytes of the file.
    """
    binary_blob = ""
    with open(file_path, "rb") as f:
        binary_blob = f.read()
    return bytearray(binary_blob)


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
    """Command line argument parsing.

    Args:
        argv: The command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse")
    parser.add_argument(
        "-c",
        "--capture_config",
        dest="cfg",
        type=ap_check_file_exists,
        required=True,
        help="Path of the attack config file",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project",
        type=ap_check_dir_exists,
        required=True,
        help="Path of the output project directory",
    )
    parser.add_argument(
        "-n",
        "--note",
        dest="notes",
        type=str,
        required=False,
        help="Notes to be stored in the project database",
    )
    parser.add_argument(
        "-b",
        "--save_bitstream",
        dest="save_bitstream",
        type=bool,
        required=False,
        help="Save bitstream into project database",
    )
    parser.add_argument(
        "-f",
        "--save_binary",
        dest="save_binary",
        type=bool,
        required=False,
        help="Save binary into project database",
    )

    args = parser.parse_args(argv)

    return args
