# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the Ibex SCA application on OpenTitan.

Communication with OpenTitan happens over the uJson command interface.
"""
import json
import time
from typing import Optional


class OTIbex:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        if protocol == "simpleserial":
            raise RuntimeError("Error: Simpleserial not supported!")

    def _ujson_ibex_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("IbexSca").encode("ascii"))

    def ibex_sca_read_response(self, num_attempts: Optional[int] = 100):
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
        """ Initializes the Ibex SCA tests on the target.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
        self.target.write(json.dumps(parameters).encode("ascii"))

    def ibex_sca_register_file_read_batch_random(self, num_segments: int):
        """ Start ibex.sca.register_file_read_batch_random test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFReadBatchRandom command.
        self.target.write(json.dumps("RFReadBatchRandom").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_read_random(self, data: list[int]):
        """ Start ibex.sca.register_file_read_random test.
        Args:
            data: The data that is first written into the RF and then read back.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFRead command.
        self.target.write(json.dumps("RFRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_read_batch_fvsr(self, data: int, num_segments: int):
        """ Start ibex.sca.register_file_read_batch_fvsr test.
        Args:
            data: The data that is first written into the RF and then read back.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFReadBatchFvsr command.
        self.target.write(json.dumps("RFReadBatchFvsr").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_read_fvsr(self, data: list[int]):
        """ Start ibex.sca.register_file_read_fvsr test.
        Args:
            data: The data that is first written into the RF and then read back.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFRead command.
        self.target.write(json.dumps("RFRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_write_batch_random(self, num_segments: int):
        """ Start ibex.sca.register_file_write_batch_random test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFWriteBatchRandom command.
        self.target.write(json.dumps("RFWriteBatchRandom").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_write_random(self, data: list[int]):
        """ Start ibex.sca.register_file_write_random test.
        Args:
            data: The data that is written into the RF.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFWrite command.
        self.target.write(json.dumps("RFWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_write_batch_fvsr(self, data: int, num_segments: int):
        """ Start ibex.sca.register_file_write_batch_fvsr test.
        Args:
            data: The data that is written into the RF.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFWriteBatchFvsr command.
        self.target.write(json.dumps("RFWriteBatchFvsr").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_register_file_write_fvsr(self, data: list[int]):
        """ Start ibex.sca.register_file_write_fvsr test.
        Args:
            data: The data that is written into the RF.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFWrite command.
        self.target.write(json.dumps("RFWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_batch_random(self, num_segments: int):
        """ Start ibex.sca.tl_write_batch_random test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWriteBatchRandom command.
        self.target.write(json.dumps("TLWriteBatchRandom").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_batch_random_fix_address(self, num_segments: int):
        """ Start ibex.sca.tl_write_batch_random_fix_address test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWriteBatchRandomFixAddress command.
        self.target.write(json.dumps("TLWriteBatchRandomFixAddress").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_random(self, data: list[int]):
        """ Start ibex.sca.tl_write_random test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWrite command.
        self.target.write(json.dumps("TLWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_batch_fvsr(self, data: int, num_segments: int):
        """ Start ibex.sca.tl_write_batch_fvsr test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWriteBatchFvsr command.
        self.target.write(json.dumps("TLWriteBatchFvsr").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_batch_fvsr_fix_address(self, data: int,
                                                 num_segments: int):
        """ Start ibex.sca.tl_write_batch_fvsr_fix_address test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWriteBatchFvsrFixAddress command.
        self.target.write(json.dumps("TLWriteBatchFvsrFixAddress").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_write_fvsr(self, data: list[int]):
        """ Start ibex.sca.tl_write_fvsr test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWrite command.
        self.target.write(json.dumps("TLWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_batch_random(self, num_segments: int):
        """ Start ibex.sca.tl_read_batch_random test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLReadBatchRandom command.
        self.target.write(json.dumps("TLReadBatchRandom").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_batch_random_fix_address(self, num_segments: int):
        """ Start ibex.sca.tl_read_batch_random_fix_address test.
        Args:
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLReadBatchRandomFixAddress command.
        self.target.write(json.dumps("TLReadBatchRandomFixAddress").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_random(self, data: list[int]):
        """ Start ibex.sca.tl_read_random test.
        Args:
            num_iterations: The number of iterations the RF is written.
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLRead command.
        self.target.write(json.dumps("TLRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_batch_fvsr(self, data: int, num_segments: int):
        """ Start ibex.sca.tl_read_batch_fvsr test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLReadBatchFvsr command.
        self.target.write(json.dumps("TLReadBatchFvsr").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_batch_fvsr_fix_address(self, data: int,
                                                num_segments: int):
        """ Start ibex.sca.tl_read_batch_fvsr_fix_address test.
        Args:
            data: The data that is written into the SRAM over Tl-UL.
            num_segments: The number of iterations.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLReadBatchFvsrFixAddress command.
        self.target.write(json.dumps("TLReadBatchFvsrFixAddress").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_segments, "fixed_data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_sca_tl_read_fvsr(self, data: list[int]):
        """ Start ibex.sca.tl_read_fvsr test.
        Args:
            num_iterations: The number of iterations the RF is written.
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLRead command.
        self.target.write(json.dumps("TLRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"data": data}
        self.target.write(json.dumps(data).encode("ascii"))

    def start_test(self, testname: str, arg1 = None, arg2 = None) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            testname: The test to start
            arg1: The first argument passed to the test.
            arg2: The second argument passed to the test.
        """
        test_function = getattr(self, testname)
        if arg1 is not None and arg2 is None:
            test_function(arg1)
        elif arg2 is not None:
            test_function(arg1, arg2)
        else:
            test_function()

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from Ibex SCA framework.
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
