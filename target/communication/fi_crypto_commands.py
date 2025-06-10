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

    def init(self, icache_disable: bool, dummy_instr_disable: bool,
             enable_jittery_clock: bool, enable_sram_readback: bool) -> list:
        """ Initialize the Crypto FI code on the chip.
        Args:
            icache_disable: If true, disable the iCache. If false, use default config
                            set in ROM.
            dummy_instr_disable: If true, disable the dummy instructions. If false,
                                 use default config set in ROM.
            enable_jittery_clock: If true, enable the jittery clock.
            enable_sram_readback: If true, enable the SRAM readback feature.
        Returns:
            The device ID and countermeasure config of the device.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        # Configure device and countermeasures.
        time.sleep(0.01)
        data = {"icache_disable": icache_disable, "dummy_instr_disable": dummy_instr_disable,
                "enable_jittery_clock": enable_jittery_clock,
                "enable_sram_readback": enable_sram_readback}
        self.target.write(json.dumps(data).encode("ascii"))
        # Read back device ID and countermeasure configuration from device.
        device_config = self.read_response(max_tries=30)
        # Read flash owner page.
        device_config += self.read_response(max_tries=30)
        # Read boot log.
        device_config += self.read_response(max_tries=30)
        # Read boot measurements.
        device_config += self.read_response(max_tries=30)
        # Read pentest framework version.
        device_config += self.read_response(max_tries=30)
        return device_config

    def crypto_shadow_reg_access(self) -> None:
        """ Starts the crypto.fi.shadow_reg_access test.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # ShadowRegAccess command.
        time.sleep(0.01)
        self.target.write(json.dumps("ShadowRegAccess").encode("ascii"))

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
                "static_trigger": False, "squeeze_trigger": False}
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
                "static_trigger": False, "squeeze_trigger": False}
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
                "static_trigger": False, "squeeze_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_kmac_static(self) -> None:
        """ Starts the crypto.fi.kmac_static test.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Kmac command.
        time.sleep(0.01)
        self.target.write(json.dumps("Kmac").encode("ascii"))
        # Mode payload.
        time.sleep(0.01)
        mode = {"key_trigger": False, "absorb_trigger": False,
                "static_trigger": True, "squeeze_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_sha256_start(self) -> None:
        """ Starts the crypto.fi.sha256_start test with a hardcoded msg of 0.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Sha256 command.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha256").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"message": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.target.write(json.dumps(data).encode("ascii"))
        time.sleep(0.01)
        # Trigger payload.
        time.sleep(0.01)
        mode = {"start_trigger": True, "msg_trigger": False, "process_trigger": False,
                "finish_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_sha256_msg(self) -> None:
        """ Starts the crypto.fi.sha256_msg test with a hardcoded msg of 0.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Sha256 command.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha256").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"message": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.target.write(json.dumps(data).encode("ascii"))
        time.sleep(0.01)
        # Trigger payload.
        time.sleep(0.01)
        mode = {"start_trigger": False, "msg_trigger": True, "process_trigger": False,
                "finish_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_sha256_process(self) -> None:
        """ Starts the crypto.fi.sha256_process test with a hardcoded msg of 0.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Sha256 command.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha256").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"message": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.target.write(json.dumps(data).encode("ascii"))
        time.sleep(0.01)
        # Trigger payload.
        time.sleep(0.01)
        mode = {"start_trigger": False, "msg_trigger": False, "process_trigger": True,
                "finish_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def crypto_sha256_finish(self) -> None:
        """ Starts the crypto.fi.sha256_finish test with a hardcoded msg of 0.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Sha256 command.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha256").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"message": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.target.write(json.dumps(data).encode("ascii"))
        time.sleep(0.01)
        # Trigger payload.
        time.sleep(0.01)
        mode = {"start_trigger": False, "msg_trigger": False, "process_trigger": False,
                "finish_trigger": True}
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
