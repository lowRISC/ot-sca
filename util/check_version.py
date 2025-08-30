# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import chipwhisperer as cw


def check_cw(cw_version_exp: str) -> None:
    """Check ChipWhisperer API version.

    Read CW API version and compare against expected version.

    Args:
        cw_version_exp: Expected CW version.

    Returns:
        Raises a runtime error on a mismatch.
    """
    cw_version = cw.__version__
    if cw_version != cw_version_exp:
        raise RuntimeError(
            f"Please update the Python requirements. CW version: \
            {cw_version}, expected CW version: {cw_version_exp}"
        )  # noqa: E501


def check_husky(husky_fw_exp: str, sn: Optional[str] = None) -> None:
    """Check ChipWhisperer Husky firmware version.

    Read CW Husky FW version and compare against expected version.

    Args:
        husky_fw_exp: Expected CW Husky FW version.

    Returns:
        Raises a runtime error on a mismatch.
    """
    if sn is not None:
        husky_fw = cw.scope(sn=str(sn)).fw_version_str
    else:
        husky_fw = cw.scope().fw_version_str
    if husky_fw != husky_fw_exp:
        raise RuntimeError(
            f"Please update the Husky firmware. FW version: {husky_fw}, \
                expected FW version: {husky_fw_exp}"
        )  # noqa: E501
