# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the EDN SCA application on OpenTitan.

Communication with OpenTitan happens over the uJson command interface.
"""
import json
import time
from typing import Optional


class OTEDN:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        if protocol == "simpleserial":
            raise RuntimeError("Error: Simpleserial not supported!")

    def _ujson_edn_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("EdnSca").encode("ascii"))
        time.sleep(0.01)

    def edn_sca_read_response(self, num_attempts: Optional[int] = 100):
        """ Reads back the "result" response from the device.
        """
        read_counter = 0
        while read_counter < num_attempts:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    if "result" in json_string:
                        return json.loads(json_string)["result"]
                except Exception:
                    raise Exception("Acknowledge error: Device and host not in sync")
            else:
                read_counter += 1
        raise Exception("Acknowledge error: Device and host not in sync")

    def init(self):
        """ Initializes the EDN SCA tests on the target.
        """
        # EdnSca command.
        self._ujson_edn_sca_cmd()
        # Init the EDN SCA tests.
        self.target.write(json.dumps("Init").encode("ascii"))

    def edn_sca_bus_data(self, init_seed: list[int],
                         reseed: list[int]):
        """ Start edn.sca.bus_data test.
        Args:
            init_seed: The initial seed.
            reseed: The reseed.
        """
        # EdnSca command.
        self._ujson_edn_sca_cmd()
        # BusData command.
        self.target.write(json.dumps("BusData").encode("ascii"))
        # Seed payload.
        time.sleep(0.01)
        data = {"init_seed": init_seed, "reseed": reseed}
        self.target.write(json.dumps(data).encode("ascii"))

    def edn_sca_bus_data_batch_random(self, num_segments: int):
        """ Start edn.sca.bus_data_batch_random test.
        Args:
            num_segments: The number of iterations.
        """
        # EdnSca command.
        self._ujson_edn_sca_cmd()
        # BusDataBatchRandom command.
        self.target.write(json.dumps("BusDataBatchRandom").encode("ascii"))
        # num_segments payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def edn_sca_bus_data_batch_fvsr(self, init_seed: list[int],
                                    reseed: list[int], num_segments: int):
        """ Start edn.sca.bus_data_batch_fvsr test.
        Args:
            init_seed: The initial seed.
            reseed: The reseed.
            num_segments: The number of iterations.
        """
        # EdnSca command.
        self._ujson_edn_sca_cmd()
        # BusDataBatchFvsrcommand.
        self.target.write(json.dumps("BusDataBatchFvsr").encode("ascii"))
        # Seed payload.
        time.sleep(0.01)
        data = {"init_seed": init_seed, "reseed": reseed,
                "num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def start_test(self, testname: str, arg1 = None, arg2 = None,
                   arg3 = None) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            testname: The test to start
            arg1: The first argument passed to the test.
            arg2: The second argument passed to the test.
            arg3: The third argument passed to the test.
        """
        test_function = getattr(self, testname)
        if arg1 is not None and arg2 is None:
            test_function(arg1)
        elif arg2 is not None and arg3 is None:
            test_function(arg1, arg2)
        elif arg3 is not None:
            test_function(arg1, arg2, arg3)
        else:
            test_function()
