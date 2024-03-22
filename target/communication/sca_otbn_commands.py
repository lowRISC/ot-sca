# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the SHA3 SCA application on OpenTitan.

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

    def init(self):
        """ Initializes OTBN on the target.
        """
        if not self.simple_serial:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Init command.
            self.target.write(json.dumps("Init").encode("ascii"))

    def write_keygen_seed(self, seed):
        """ Write the seed used for the keygen app.
        Args:
            seed: Byte array containing the seed.
        """
        # OtbnSca command.
        self._ujson_otbn_sca_cmd()
        # Ecc256SetSeed command.
        time.sleep(0.01)
        self.target.write(json.dumps("Ecc256SetSeed").encode("ascii"))
        # Seed payload.
        time.sleep(0.01)
        seed_int = [x for x in seed]
        seed_data = {"seed": seed_int}
        self.target.write(json.dumps(seed_data).encode("ascii"))

    def write_keygen_key_constant_redundancy(self, const):
        """ Write the constant redundancy value for the keygen app.
        Args:
            const: Byte array containing the redundancy value.
            const_length: The length of the constant.
        """
        # OtbnSca command.
        self._ujson_otbn_sca_cmd()
        # Ecc256SetC command.
        self.target.write(json.dumps("Ecc256SetC").encode("ascii"))
        # Constant payload.
        time.sleep(0.01)
        const_int = [x for x in const]
        const_data = {"constant": const_int}
        self.target.write(json.dumps(const_data).encode("ascii"))

    def config_keygen_masking(self, masks_on):
        """ Disable/enable masking.
        Args:
            masks_on: Boolean value.
        """
        # OtbnSca command.
        self._ujson_otbn_sca_cmd()
        # Ecc256EnMasks command.
        self.target.write(json.dumps("Ecc256EnMasks").encode("ascii"))
        # Enable/disable masks payload.
        time.sleep(0.01)
        mask = {"en_masks": masks_on}
        self.target.write(json.dumps(mask).encode("ascii"))

    def start_keygen_batch(self, test_type, num_segments):
        """ Start the keygen app in batch mode.
        Args:
            test_type: String selecting the test type (KEY or SEED).
            num_segments: Number of keygen executions to perform.
        """
        # OtbnSca command.
        self._ujson_otbn_sca_cmd()
        if test_type == 'KEY':
            # Ecc256EcdsaKeygenFvsrKeyBatch command.
            self.target.write(json.dumps("Ecc256EcdsaKeygenFvsrKeyBatch").encode("ascii"))
        else:
            # Ecc256EcdsaKeygenFvsrSeedBatch command.
            self.target.write(json.dumps("Ecc256EcdsaKeygenFvsrSeedBatch").encode("ascii"))
        # Num traces payload.
        time.sleep(0.01)
        num_segments_data = {"num_traces": num_segments}
        self.target.write(json.dumps(num_segments_data).encode("ascii"))

    def read_batch_digest(self):
        """ Read d0 from the device.

        Returns:
            The received output.
        """
        while True:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    return json.loads(json_string)["batch_digest"]
                except Exception:
                    pass  # noqa: E302
