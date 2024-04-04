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


class OTOTBNVERT:
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

    def choose_otbn_app(self, app):
        """ Select the OTBN application.
        Args:
            app: OTBN application
        """
        # Select the otbn app on the device (0 -> keygen, 1 -> modinv).
        app_value = 0x00
        if app == 'modinv':
            app_value = 0x01
        if self.simple_serial:
            self.target.write(cmd="a", data=bytearray([app_value]))
        else:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Ecc256AppSelect command.
            self.target.write(json.dumps("Ecc256AppSelect").encode("ascii"))
            # App payload.
            time.sleep(0.01)
            app_select = {"app": app_value}
            self.target.write(json.dumps(app_select).encode("ascii"))
            time.sleep(0.01)

    def write_keygen_seed(self, seed, seed_length: Optional[int] = 40):
        """ Write the seed used for the keygen app.
        Args:
            seed: Byte array containing the seed.
            seed_length: The length of the seed.
        """
        if self.simple_serial:
            self.target.write(cmd='x', data=seed)
        else:
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

    def write_keygen_key_constant_redundancy(self, const, const_length: Optional[int] = 40):
        """ Write the constant redundancy value for the keygen app.
        Args:
            const: Byte array containing the redundancy value.
            const_length: The length of the constant.
        """
        if self.simple_serial:
            self.target.write(cmd="c", data=const)
        else:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Ecc256SetC command.
            self.target.write(json.dumps("Ecc256SetC").encode("ascii"))
            # Constant payload.
            time.sleep(0.01)
            const_int = [x for x in const]
            const_data = {"constant": const_int, "constant_len": const_length}
            self.target.write(json.dumps(const_data).encode("ascii"))

    def config_keygen_masking(self, off):
        """ Disable/enable masking.
        Args:
            off: Boolean value.
        """
        # Enable/disable masking.
        off_int = 0x01
        if off is True:
            off_int = 0x00
        if self.simple_serial:
            self.target.write(cmd="m", data=bytearray([off_int]))
        else:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Ecc256EnMasks command.
            self.target.write(json.dumps("Ecc256EnMasks").encode("ascii"))
            # Enable/disable masks payload.
            time.sleep(0.01)
            mask = {"en_masks": off_int}
            self.target.write(json.dumps(mask).encode("ascii"))

    def start_keygen(self, mask):
        """ Write the seed mask and start the keygen app.
        Args:
            mask: Byte array containing the mask value.
        """
        if self.simple_serial:
            # Send the mask and start the keygen operation.
            self.target.write(cmd='k', data=mask)
        else:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Ecc256EcdsaSecretKeygen command.
            self.target.write(json.dumps("Ecc256EcdsaSecretKeygen").encode("ascii"))
            # Mask payload.
            time.sleep(0.01)
            mask_int = [x for x in mask]
            mask_data = {"mask": mask_int[0:20]}
            self.target.write(json.dumps(mask_data).encode("ascii"))
            time.sleep(0.01)
            mask_data = {"mask": mask_int[20:41]}
            self.target.write(json.dumps(mask_data).encode("ascii"))

    def start_modinv(self, scalar_k0, scalar_k1, k_length: Optional[int] = 80):
        """ Write the two scalar shares and start the modinv app.
        Args:
            scalar_k0: Byte array containing the first scalar share.
            scalar_k1: Byte array containing the second scalar share.
            k_length: The length of the scalar shares.
        """
        if self.simple_serial:
            # Start modinv device computation.
            self.target.write(cmd='q', data=(scalar_k0 + scalar_k1))
        else:
            # OtbnSca command.
            self._ujson_otbn_sca_cmd()
            # Ecc256Modinv command.
            self.target.write(json.dumps("Ecc256Modinv").encode("ascii"))
            # Scalar payload.
            time.sleep(0.01)
            scalar_int = [x for x in (scalar_k0 + scalar_k1)]
            scalar_data = {"k": scalar_int[0:20]}
            self.target.write(json.dumps(scalar_data).encode("ascii"))
            time.sleep(0.01)
            scalar_data = {"k": scalar_int[20:40]}
            self.target.write(json.dumps(scalar_data).encode("ascii"))
            time.sleep(0.01)
            scalar_data = {"k": scalar_int[40:60]}
            self.target.write(json.dumps(scalar_data).encode("ascii"))
            time.sleep(0.01)
            scalar_data = {"k": scalar_int[60:80]}
            self.target.write(json.dumps(scalar_data).encode("ascii"))

    def start_keygen_batch(self, test_type, num_segments):
        """ Start the keygen app in batch mode.
        Args:
            test_type: String selecting the test type (KEY or SEED).
            num_segments: Number of keygen executions to perform.
        """
        if test_type == 'KEY':
            if self.simple_serial:
                # Start batch keygen.
                self.target.write(cmd="e", data=num_segments)
            else:
                # OtbnSca command.
                self._ujson_otbn_sca_cmd()
                # Ecc256EcdsaKeygenFvsrKeyBatch command.
                self.target.write(json.dumps("Ecc256EcdsaKeygenFvsrKeyBatch").encode("ascii"))
                time.sleep(0.01)
                num_segments_data = {"num_traces": num_segments}
                self.target.write(json.dumps(num_segments_data).encode("ascii"))
        else:
            if self.simple_serial:
                self.target.write(cmd="b", data=num_segments)
            else:
                # OtbnSca command.
                self._ujson_otbn_sca_cmd()
                # Ecc256EcdsaKeygenFvsrSeedBatch command.
                self.target.write(json.dumps("Ecc256EcdsaKeygenFvsrSeedBatch").encode("ascii"))
                time.sleep(0.01)
                num_segments_data = {"num_traces": num_segments}
                self.target.write(json.dumps(num_segments_data).encode("ascii"))

    def read_alpha(self, kalpha_inv_length: int, alpha_length: int):
        """ Read alpha & kalpha_inv from the device.
        Args:
            kalpha_inv_length: Number of bytes to read for kalpha_inv.
            alpha_length: Number of bytes to read for alpha.
        Returns:
            The received output.
        """
        if self.simple_serial:
            kalpha_inv = self.target.read("r", kalpha_inv_length, ack=False)
            alpha = self.target.read("r", alpha_length, ack=False)
            return kalpha_inv, alpha
        else:
            while True:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        kalpha_inv = json.loads(json_string)["alpha_inv"]
                        alpha = json.loads(json_string)["alpha"]
                        return kalpha_inv, alpha
                    except Exception:
                        pass  # noqa: E302

    def read_seeds(self, seed_bytes: int):
        """ Read d0 and d1 from the device.
        Args:
            seed_bytes: Number of bytes to read for kalpha_inv.
            alpha_length: Number of bytes to read for alpha.
        Returns:
            The received output.
        """
        if self.simple_serial:
            share0 = self.target.read("r", seed_bytes, ack=False)
            share1 = self.target.read("r", seed_bytes, ack=False)
            if share0 is None:
                raise RuntimeError('Random share0 is none')
            if share1 is None:
                raise RuntimeError('Random share1 is none')

            return share0, share1
        else:
            d0 = None
            d1 = None
            while True:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    if "d0" in json_string:
                        d0 = json.loads(json_string)["d0"]
                    elif "d1" in json_string:
                        d1 = json.loads(json_string)["d1"]
                    if d0 is not None and d1 is not None:
                        return d0, d1
