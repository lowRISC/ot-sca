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
        self.target.write(json.dumps("CryptoFi").encode("ascii"))
        time.sleep(0.01)

    def init(self):
        """ Initialize the Crypto FI code on the chip.
        """
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
        self.target.write(json.dumps(parameters).encode("ascii"))

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
        

    def read_response(self) -> str:
        """ Read response from Crypto FI framework.
        Returns:
            The JSON response of OpenTitan.
        """
        while True:
            read_line = str(self.target.readline())
            if len(read_line) < 5:
                break
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
        return ""
    
    def crypto_sha2(self, msg, trigger) -> None:
        # CryptoFi command.
        self._ujson_crypto_cmd()
        # Sha2 command.
        self.target.write(json.dumps("Sha256").encode("ascii"))
        time.sleep(0.01)
        data = {"message": msg}
        self.target.write(json.dumps(data).encode("ascii"))
        time.sleep(0.01)
        if trigger == 0:
            mode = {"start_trigger": True, "msg_trigger": False, "process_trigger" : False,
                    "finish_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "msg_trigger": True, "process_trigger" : False,
                    "finish_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "msg_trigger": False, "process_trigger" : True,
                    "finish_trigger": False}
        else:
            mode = {"start_trigger": False, "msg_trigger": False, "process_trigger" : False,
                    "finish_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

        
    def read_digest(self):
        """ Read tag from OpenTitan HMAC.

        Returns:
            The received tag.
        """
        while True:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    tag = json.loads(json_string)["digest"]
                    return tag
                except Exception:
                    pass  # noqa: E302