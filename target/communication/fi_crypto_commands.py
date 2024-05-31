# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Crypto FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


from target.communication.otfi import OTFI


class OTFICrypto(OTFI):
    def __init__(self, target) -> None:
        super().__init__(target, "Crypto")

    def crypto_shadow_reg_access(self) -> None:
        """ Starts the crypto.fi.shadow_reg_access test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # ShadowRegAccess command.
        time.sleep(0.01)
        self.target.write(json.dumps("ShadowRegAccess").encode("ascii"))

    def crypto_aes_key(self) -> None:
        """ Starts the crypto.fi.aes_key test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Aes command.
        time.sleep(0.01)
        self.target.write(json.dumps("Aes").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": True, "plaintext_trigger": False,
                "encrypt_trigger": False, "ciphertext_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_aes_plaintext(self) -> None:
        """ Starts the crypto.fi.aes_plaintext test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Aes command.
        time.sleep(0.01)
        self.target.write(json.dumps("Aes").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "plaintext_trigger": True,
                "encrypt_trigger": False, "ciphertext_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_aes_encrypt(self) -> None:
        """ Starts the crypto.fi.aes_encrypt test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Aes command.
        time.sleep(0.01)
        self.target.write(json.dumps("Aes").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "plaintext_trigger": False,
                "encrypt_trigger": True, "ciphertext_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_aes_ciphertext(self) -> None:
        """ Starts the crypto.fi.aes_ciphertext test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Aes command.
        time.sleep(0.01)
        self.target.write(json.dumps("Aes").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "plaintext_trigger": False,
                "encrypt_trigger": False, "ciphertext_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_key(self) -> None:
        """ Starts the crypto.fi.kmac_key test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": True, "absorb_trigger": False,
                "static_trigger": False, "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_absorb(self) -> None:
        """ Starts the crypto.fi.kmac_absorb test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": True,
                "static_trigger": False, "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_squeeze(self) -> None:
        """ Starts the crypto.fi.kmac_squeeze test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": False,
                "static_trigger": False, "squeeze_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_static(self) -> None:
        """ Starts the crypto.fi.kmac_static test.
        """
        # CryptoFi command.
        self._ujson_fi_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": False,
                "static_trigger": True, "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))
