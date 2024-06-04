# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Crypto FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import copy
import json
import time
from typing import Optional


from target.communication.otfi import OTFI
from target.communication.otfi_test import OTFITest


MODES = {
    "aes": {
        "key_trigger": False, "plaintext_trigger": False,
        "encrypt_trigger": False, "ciphertext_trigger": False
    },
    "kmac": {
        "key_trigger": False, "absorb_trigger": False,
        "static_trigger": False, "squeeze_trigger": False
    },
}


def _get_mode(ip, mode_id):
    assert ip in MODES, f"IP {ip} not in MODES ({MODES})"
    mode = copy.deepcopy(MODES[ip])
    assert mode_id in mode, f"Mode id {mode_id} not in {ip} mode ({mode})"
    mode[mode_id] = True
    return mode


class OTFICrypto(OTFI):
    TESTS = [
        OTFITest("shadow_reg_access"),
        OTFITest("aes_key", "Aes", _get_mode("aes", "key_trigger")),
        OTFITest("aes_plaintext", "Aes", _get_mode("aes", "plaintext_trigger")),
        OTFITest("aes_encrypt", "Aes", _get_mode("aes", "encrypt_trigger")),
        OTFITest("aes_ciphertext", "Aes", _get_mode("aes", "ciphertext_trigger")),
        OTFITest("kmac_key", "Kmac", _get_mode("kmac", "key_trigger")),
        OTFITest("kmac_absorb", "Kmac", _get_mode("kmac", "absorb_trigger")),
        OTFITest("kmac_static", "Kmac", _get_mode("kmac", "static_trigger")),
        OTFITest("kmac_squeeze", "Kmac", _get_mode("kmac", "squeeze_trigger")),
    ]

    def __init__(self, target) -> None:
        super().__init__(target, "Crypto")
