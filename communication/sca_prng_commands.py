# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the PRNG SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTPRNG:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_prng_sca_cmd(self):
        time.sleep(0.01)
        self.target.write(json.dumps("PrngSca").encode("ascii"))

    def seed_prng(self, seed, seed_length: Optional[int] = 4):
        """ Seed the PRNG.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="s", data=seed)
        else:
            # PrngSca command.
            self._ujson_prng_sca_cmd()
            # SingleEncrypt command.
            time.sleep(0.01)
            self.target.write(json.dumps("SeedPrng").encode("ascii"))
            # Text payload.
            seed_int = [x for x in seed]
            seed_data = {"seed": seed_int, "seed_length": seed_length}
            self.target.write(json.dumps(seed_data).encode("ascii"))
