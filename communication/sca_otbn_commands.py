# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the OTBN SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTOTBN:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_otbn_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("OtbnSca").encode("ascii"))

    def init(self, icache_disable: bool, dummy_instr_disable: bool):
        """ Initializes OTBN on the target.
        Args:
            icache_disable: If true, disable the iCache. If false, use default config
                            set in ROM.
            dummy_instr_disable: If true, disable the dummy instructions. If false,
                                 use default config set in ROM.
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Init the OTBN core.
            self.target.write(json.dumps("Init").encode("ascii"))
            parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
            self.target.write(json.dumps(parameters).encode("ascii"))

    def init_keymgr(self):
        """ Initializes the key manager for OTBN on the target.
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Init the key manager.
            self.target.write(json.dumps("InitKeyMgr").encode("ascii"))

    def key_sideload_fvsr(self, seed: int):
        """ Starts the key sidloading FvsR test on OTBN.
        Args:
            seed: The fixed seed used by the key manager.
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Start the KeySideloadFvsr test.
            self.target.write(json.dumps("KeySideloadFvsr").encode("ascii"))
            time.sleep(0.01)
            seed_data = {"fixed_seed": seed}
            self.target.write(json.dumps(seed_data).encode("ascii"))

    def ecdsa_p256_sign(self, masking_on: bool, msg, d0, k0):
        """ Starts the EcdsaP256Sign test on OTBN.
        Args:
            masking_on: Turn on/off masking.
            msg: Message array (8xuint32_t)
            d0: Message array (10xuint32_t)
            k0: Message array (10xuint32_t)
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Start the EcdsaP256Sign test.
            self.target.write(json.dumps("EcdsaP256Sign").encode("ascii"))
            time.sleep(0.01)
            # Configure masking.
            masks = {"en_masks": masking_on}
            self.target.write(json.dumps(masks).encode("ascii"))
            time.sleep(0.01)
            # Send msg, d0, and k0.
            data = {"msg": msg, "d0": d0, "ko": k0}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)

    def ecdsa_p256_sign_batch(self, num_traces: int, masking_on: bool, msg, d0, k0):
        """ Starts the EcdsaP256SignBatch test on OTBN.
        Args:
            num_traces: Number of batch operations.
            masking_on: Turn on/off masking.
            msg: Message array (8xuint32_t)
            d0: Message array (10xuint32_t)
            k0: Message array (10xuint32_t)
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Start the EcdsaP256SignBatch test.
            self.target.write(json.dumps("EcdsaP256Sign").encode("ascii"))
            time.sleep(0.01)
            # Configure number of traces.
            num_traces = {"num_traces": num_traces}
            self.target.write(json.dumps(num_traces).encode("ascii"))
            time.sleep(0.01)
            # Configure masking.
            masks = {"en_masks": masking_on}
            self.target.write(json.dumps(masks).encode("ascii"))
            time.sleep(0.01)
            # Send msg, d0, and k0.
            data = {"msg": msg, "d0": d0, "ko": k0}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)

    def ecdsa_p256_sign_batch_fvsr(self, num_traces: int, masking_on: bool, msg, d0, k0):
        """ Starts the EcdsaP256SignFvsrBatch test on OTBN.
        Args:
            num_traces: Number of batch operations.
            masking_on: Turn on/off masking.
            msg: Message array (8xuint32_t)
            d0: Message array (10xuint32_t)
            k0: Message array (10xuint32_t)
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Start the EcdsaP256SignBatch test.
            self.target.write(json.dumps("EcdsaP256SignFvsrBatch").encode("ascii"))
            time.sleep(0.01)
            # Configure number of traces.
            num_traces = {"num_traces": num_traces}
            self.target.write(json.dumps(num_traces).encode("ascii"))
            time.sleep(0.01)
            # Configure masking.
            masks = {"en_masks": masking_on}
            self.target.write(json.dumps(masks).encode("ascii"))
            time.sleep(0.01)
            # Send msg, d0, and k0.
            data = {"msg": msg, "d0": d0, "ko": k0}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)

    def choose_otbn_app(self, app):
        """ Select the OTBN application.
        Args:
            app: OTBN application
        """
        if self.simple_serial:
            # Select the otbn app on the device (0 -> keygen, 1 -> modinv).
            if app == 'keygen':
                self.target.write(cmd="a", data=bytearray([0x00]))
            if app == 'modinv':
                self.target.write(cmd="a", data=bytearray([0x01]))

    def write_batch_prng_seed(self, seed):
        """ Seed the PRNG.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="s", data=seed)

    def write_keygen_seed(self, seed):
        """ Write the seed used for the keygen app.
        Args:
            seed: byte array containing the seed.
        """
        if self.simple_serial:
            self.target.write(cmd='x', data=seed)

    def write_keygen_key_constant_redundancy(self, const):
        """ Write the constant redundancy value for the keygen app.
        Args:
            seed: byte array containing the redundancy value.
        """
        if self.simple_serial:
            self.target.write(cmd="c", data=const)

    def config_keygen_masking(self, off):
        """ Disable/enable masking.
        Args:
            off: boolean value.
        """
        if self.simple_serial:
            # Enable/disable masking.
            if off is True:
                self.target.write(cmd="m", data=bytearray([0x00]))
            else:
                self.target.write(cmd="m", data=bytearray([0x01]))

    def start_keygen(self, mask):
        """ Write the seed mask and start the keygen app.
        Args:
            mask: byte array containing the mask value.
        """
        if self.simple_serial:
            # Send the mask and start the keygen operation.
            self.target.write('k', mask)

    def start_modinv(self, scalar_k0, scalar_k1):
        """ Write the two scalar shares and start the modinv app.
        Args:
            scalar_k0: byte array containing the first scalar share.
            scalar_k1: byte array containing the second scalar share.
        """
        if self.simple_serial:
            # Start modinv device computation.
            self.target.write('q', scalar_k0 + scalar_k1)

    def start_keygen_batch(self, test_type, num_segments):
        """ Start the keygen app in batch mode.
        Args:
            test_type: string selecting the test type (KEY or SEED).
            num_segments: number of keygen executions to perform.
        """
        if self.simple_serial:
            # Start batch keygen.
            if test_type == 'KEY':
                self.target.write(cmd="e", data=num_segments)
            else:
                self.target.write(cmd="b", data=num_segments)

    def read_output(self, len_bytes):
        """ Read the output from whichever OTBN app.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received output.
        """
        if self.simple_serial:
            return self.target.read("r", len_bytes, ack=False)

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from OTBN SCA framework.
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
