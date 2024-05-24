# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Crypto FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFICrypto:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_crypto_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("CryptoFi").encode("ascii"))
        time.sleep(0.01)

    def init(self) -> None:
        """ Initialize the Crypto FI code on the chip.
        Returns:
            The device ID of the device.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        # Read back device ID from device.
        return self.read_response(max_tries=30)

    def crypto_aes_key(self) -> None:
        """ Starts the crypto.fi.aes_key test.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
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
        self._ujson_crypto_cmd()
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
        self._ujson_crypto_cmd()
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
        self._ujson_crypto_cmd()
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
        self._ujson_crypto_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": True, "absorb_trigger": False,
                "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_absorb(self) -> None:
        """ Starts the crypto.fi.kmac_absorb test.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": True,
                "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_squeeze(self) -> None:
        """ Starts the crypto.fi.kmac_squeeze test.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": False,
                "squeeze_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def start_test(self, cfg: dict) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            cfg: Config dict containing the selected test.
        """
        test_function = getattr(self, cfg["test"]["which_test"])
        test_function()

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from Crypto FI framework.
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            it += 1
        return ""
