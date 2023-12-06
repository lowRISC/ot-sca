# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for different OpenTitan ciphers.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional

import serial


class OTUART:
    def __init__(self, protocol: str, port: Optional[str] = "None",
                 baudrate: Optional[int] = 115200) -> None:
        self.uart = None
        if protocol == "ujson":
            if not port:
                raise RuntimeError("Error: No uJson port provided!")
            else:
                self.uart = serial.Serial(port)
                self.uart.baudrate = baudrate


class OTAES:
    def __init__(self, target, protocol: str,
                 port: Optional[OTUART] = None) -> None:
        self.target = target
        self.simple_serial = True
        self.port = port
        if protocol == "ujson":
            self.simple_serial = False
            # Init the AES core.
            self.port.write(json.dumps("AesSca").encode("ascii"))
            self.port.write(json.dumps("Init").encode("ascii"))

    def _ujson_aes_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands.
        time.sleep(0.01)
        self.port.write(json.dumps("AesSca").encode("ascii"))

    def key_set(self, key: list[int], key_length: Optional[int] = 16):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.simpleserial_write("k", bytearray(key))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # KeySet command.
            self.port.write(json.dumps("KeySet").encode("ascii"))
            # Key payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": key_length}
            self.port.write(json.dumps(key_data).encode("ascii"))

    def fvsr_key_set(self, key, key_length: Optional[int] = 16):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.simpleserial_write("f", bytearray(key))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeySet command.
            self.port.write(json.dumps("FvsrKeySet").encode("ascii"))
            # Key payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": key_length}
            self.port.write(json.dumps(key_data).encode("ascii"))

    def seed_lfsr(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.simpleserial_write("l", seed)
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # SeedLfsr command.
            self.port.write(json.dumps("SeedLfsr").encode("ascii"))
            # Seed payload.
            time.sleep(0.01)
            seed_data = {"seed": [x for x in seed]}
            self.port.write(json.dumps(seed_data).encode("ascii"))

    def start_fvsr_batch_generate(self):
        """Set SW PRNG to starting values for FvsR data
        generation.
        """
        command = 1
        if self.simple_serial:
            self.target.simpleserial_write("d", command.to_bytes(4, "little"))
            self.target.simpleserial_wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyStartBatchGenerate command.
            self.port.write(json.dumps("FvsrKeyStartBatchGenerate").encode("ascii"))
            # Command.
            time.sleep(0.01)
            cmd = {"data": [command]}
            self.port.write(json.dumps(cmd).encode("ascii"))

    def write_fvsr_batch_generate(self, num_segments):
        """ Generate random plaintexts for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.simpleserial_write("g", num_segments)
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyBatchGenerate command.
            self.port.write(json.dumps("FvsrKeyBatchGenerate").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.port.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_alternative_encrypt(self, num_segments):
        """ Start encryption for batch (alternative).
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.simpleserial_write("a", num_segments)
            self.target.simpleserial_wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # BatchEncrypt command.
            self.port.write(json.dumps("BatchAlternativeEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.port.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_encrypt(self, num_segments):
        """ Start encryption for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.simpleserial_write("a", num_segments)
            self.target.simpleserial_wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # BatchEncrypt command.
            self.port.write(json.dumps("BatchEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.port.write(json.dumps(num_encryption_data).encode("ascii"))

    def fvsr_key_batch_encrypt(self, num_segments):
        """ Start batch encryption for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.simpleserial_write("e", num_segments)
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyBatchEncrypt command.
            self.port.write(json.dumps("FvsrKeyBatchEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.port.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_plaintext_set(self, text, text_length: Optional[int] = 16):
        """ Write plaintext to OpenTitan AES.

        This command is designed to set the initial plaintext for
        batch_alternative_encrypt.

        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.simpleserial_write("i", bytearray(text))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # BatchPlaintextSet command.
            self.port.write(json.dumps("BatchPlaintextSet").encode("ascii"))
            # Text payload.
            time.sleep(0.01)
            text_int = [x for x in text]
            text_data = {"text": text_int, "text_length": text_length}
            self.port.write(json.dumps(text_data).encode("ascii"))

    def single_encrypt(self, text: list[int], text_length: Optional[int] = 16):
        """ Write plaintext to OpenTitan AES & start encryption.
        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.simpleserial_write("p", bytearray(text))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # SingleEncrypt command.
            self.port.write(json.dumps("SingleEncrypt").encode("ascii"))
            # Text payload.
            time.sleep(0.01)
            text_data = {"text": text, "text_length": text_length}
            self.port.write(json.dumps(text_data).encode("ascii"))

    def read_ciphertext(self, len_bytes):
        """ Read ciphertext from OpenTitan AES.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received ciphertext.
        """
        if self.simple_serial:
            response_byte = self.target.simpleserial_read("r", len_bytes, ack=False)
            # Convert response into int array.
            return [x for x in response_byte]
        else:
            while True:
                read_line = str(self.port.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        ciphertext = json.loads(json_string)["ciphertext"]
                        return ciphertext[0:len_bytes]
                    except Exception:
                        pass  # noqa: E302


class OTPRNG:
    def __init__(self, target, protocol: str,
                 port: Optional[OTUART] = None) -> None:
        self.target = target
        self.simple_serial = True
        self.port = port
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_prng_sca_cmd(self):
        time.sleep(0.01)
        self.port.write(json.dumps("PrngSca").encode("ascii"))

    def seed_prng(self, seed, seed_length: Optional[int] = 4):
        """ Seed the PRNG.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.simpleserial_write("s", seed)
        else:
            # PrngSca command.
            self._ujson_prng_sca_cmd()
            # SingleEncrypt command.
            time.sleep(0.01)
            self.port.write(json.dumps("SeedPrng").encode("ascii"))
            # Text payload.
            seed_int = [x for x in seed]
            seed_data = {"seed": seed_int, "seed_length": seed_length}
            self.port.write(json.dumps(seed_data).encode("ascii"))


class OTKMAC:
    def __init__(self, target, protocol: str,
                 port: Optional[OTUART] = None) -> None:
        self.target = target
        self.simple_serial = True
        self.port = port
        if protocol == "ujson":
            self.simple_serial = False
            # Init the KMAC core.
            self.port.write(json.dumps("KmacSca").encode("ascii"))
            self.port.write(json.dumps("Init").encode("ascii"))

    def _ujson_kmac_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands.
        time.sleep(0.01)
        self.port.write(json.dumps("KmacSca").encode("ascii"))

    def write_key(self, key: list[int]):
        """ Write the key to KMAC.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.simpleserial_write("k", bytearray(key))
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SetKey command.
            self.port.write(json.dumps("SetKey").encode("ascii"))
            # Key payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": 16}
            self.port.write(json.dumps(key_data).encode("ascii"))

    def fvsr_key_set(self, key: list[int], key_length: Optional[int] = 16):
        """ Write the fixed key to KMAC.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.simpleserial_write("f", bytearray(key))
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SetKey command.
            self.port.write(json.dumps("FixedKeySet").encode("ascii"))
            # FixedKeySet payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": key_length}
            self.port.write(json.dumps(key_data).encode("ascii"))

    def write_lfsr_seed(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.simpleserial_write("l", seed)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SeedLfsr command.
            self.port.write(json.dumps("SeedLfsr").encode("ascii"))
            # Seed payload.
            time.sleep(0.01)
            seed_int = [x for x in seed]
            seed_data = {"seed": seed_int}
            self.port.write(json.dumps(seed_data).encode("ascii"))

    def absorb_batch(self, num_segments):
        """ Start absorb for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.simpleserial_write("b", num_segments)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # Batch command.
            self.port.write(json.dumps("Batch").encode("ascii"))
            # Num_segments payload.
            time.sleep(0.01)
            num_segments_data = {"data": [x for x in num_segments]}
            self.port.write(json.dumps(num_segments_data).encode("ascii"))

    def absorb(self, text, text_length: Optional[int] = 16):
        """ Write plaintext to OpenTitan KMAC & start absorb.
        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.simpleserial_write("p", text)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SingleAbsorb command.
            self.port.write(json.dumps("SingleAbsorb").encode("ascii"))
            # Msg payload.
            time.sleep(0.01)
            text_int = [x for x in text]
            text_data = {"msg": text_int, "msg_length": text_length}
            self.port.write(json.dumps(text_data).encode("ascii"))

    def read_ciphertext(self, len_bytes):
        """ Read ciphertext from OpenTitan KMAC.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received ciphertext.
        """
        if self.simple_serial:
            response_byte = self.target.simpleserial_read("r", len_bytes, ack=False)
            # Convert response into int array.
            return [x for x in response_byte]
        else:
            while True:
                read_line = str(self.port.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        batch_digest = json.loads(json_string)["batch_digest"]
                        return batch_digest[0:len_bytes]
                    except Exception:
                        pass  # noqa: E302
