# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan OTBN FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFIOtbn:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_otbn_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("OtbnFi").encode("ascii"))

    def otbn_char_dmem_access(self) -> None:
        """ Starts the otbn.fi.char.dmem.access test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharDmemAccess command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharDmemAccess").encode("ascii"))

    def otbn_char_dmem_write(self) -> None:
        """ Starts the otbn.fi.char.dmem.write test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharDmemWrite command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharDmemWrite").encode("ascii"))

    def otbn_char_rf(self) -> None:
        """ Starts the otbn.fi.char.rf test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharRF command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharRF").encode("ascii"))

    def otbn_char_beq(self) -> None:
        """ Starts the otbn.fi.char.beq test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharBeq command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharBeq").encode("ascii"))

    def otbn_char_jal(self) -> None:
        """ Starts the otbn.fi.char.jal test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharJal command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharJal").encode("ascii"))

    def otbn_char_mem(self) -> None:
        """ Starts the otbn.fi.char.mem test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharMem command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharMem").encode("ascii"))

    def otbn_char_bn_sel(self) -> None:
        """ Starts the otbn.fi.char.bn.sel test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharBnSel command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharBnSel").encode("ascii"))

    def otbn_char_bn_rshi(self) -> None:
        """ Starts the otbn.fi.char.bn.rshi test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharBnRshi command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharBnRshi").encode("ascii"))

    def otbn_char_bn_wsrr(self) -> None:
        """ Starts the otbn.fi.char.bn.wsrr test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharBnWsrr command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharBnWsrr").encode("ascii"))

    def otbn_char_lw(self) -> None:
        """ Starts the otbn.fi.char.lw test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharLw command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharLw").encode("ascii"))

    def otbn_char_unrolled_reg_op_loop(self) -> None:
        """ Starts the otbn.fi.char.unrolled.reg.op.loop test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharUnrolledRegOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUnrolledRegOpLoop").encode("ascii"))

    def otbn_char_unrolled_dmem_op_loop(self) -> None:
        """ Starts the otbn.fi.char.unrolled.dmem.op.loop test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharUnrolledDmemOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUnrolledDmemOpLoop").encode("ascii"))

    def otbn_char_hardware_reg_op_loop(self) -> None:
        """ Starts the otbn.fi.char.hardware.reg.op.loop test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharHardwareRegOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharHardwareRegOpLoop").encode("ascii"))

    def otbn_char_hardware_dmem_op_loop(self) -> None:
        """ Starts the otbn.fi.char.hardware.dmem.op.loop test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # CharMemOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharHardwareDmemOpLoop").encode("ascii"))

    def otbn_key_sideload(self) -> None:
        """ Starts the otbn.fi.key_sideload test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # KeySideload command.
        time.sleep(0.01)
        self.target.write(json.dumps("KeySideload").encode("ascii"))

    def otbn_load_integrity(self) -> None:
        """ Starts the otbn.fi.load_integrity test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # LoadIntegrity command.
        time.sleep(0.01)
        self.target.write(json.dumps("LoadIntegrity").encode("ascii"))

    def otbn_pc(self) -> None:
        """ Starts the otbn.pc test.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # PC command.
        time.sleep(0.01)
        self.target.write(json.dumps("PC").encode("ascii"))

    def init_keymgr(self, test: str) -> None:
        """ Initialize the key manager on the chip.
        Args:
            test: Name of the test. Used to determine if key manager init is
                  needed.
        """
        if "key_sideload" in test:
            # OtbnFi command.
            self._ujson_otbn_fi_cmd()
            # InitTrigger command.
            time.sleep(0.01)
            self.target.write(json.dumps("InitKeyMgr").encode("ascii"))
            time.sleep(0.5)

    def init(self, enable_icache: bool, enable_dummy_instr: bool,
             enable_jittery_clock: bool, enable_sram_readback: bool) -> list:
        """ Initialize the OTBN FI code on the chip.
        Args:
            enable_icache: If true, enable the iCache.
            enable_dummy_instr:  If true, enable the dummy instructions.
            enable_jittery_clock: If true, enable the jittery clock.
            enable_sram_readback: If true, enable the SRAM readback feature.
        Returns:
            The device ID and countermeasure config of the device.
        """
        # OtbnFi command.
        self._ujson_otbn_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        # Configure device and countermeasures.
        time.sleep(0.01)
        data = {"enable_icache": enable_icache, "enable_dummy_instr": enable_dummy_instr,
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

    def start_test(self, cfg: dict) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            cfg: Config dict containing the selected test.
        """
        test_function = getattr(self, cfg["test"]["which_test"])
        test_function()

    def write_payload(self, payload: dict) -> None:
        """ Send test payload to OpenTitan.
        Args:
            payload: The data to send to the target.
        """
        time.sleep(0.01)
        self.target.write(json.dumps(payload).encode("ascii"))

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from Otbn FI framework.
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
