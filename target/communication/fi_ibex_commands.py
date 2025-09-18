# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Ibex FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time

from target.communication import common_library


class OTFIIbex:

    def __init__(self, target) -> None:
        self.target = target

    def _ujson_ibex_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("IbexFi").encode("ascii"))

    def ibex_char_unrolled_reg_op_loop(self) -> None:
        """Starts the ibex.char.unrolled_reg_op_loop test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUnrolledRegOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUnrolledRegOpLoop").encode("ascii"))

    def ibex_char_unrolled_mem_op_loop(self) -> None:
        """Starts the ibex.char.unrolled_mem_op_loop test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUnrolledMemOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUnrolledMemOpLoop").encode("ascii"))

    def ibex_char_reg_op_loop(self) -> None:
        """Starts the ibex.char.reg_op_loop test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharRegOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharRegOpLoop").encode("ascii"))

    def ibex_char_unrolled_reg_op_loop_chain(self) -> None:
        """Starts the ibex.char.reg_op_loop_chain test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUnrolledRegOpLoopChain command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharUnrolledRegOpLoopChain").encode("ascii"))

    def ibex_char_mem_op_loop(self) -> None:
        """Starts the ibex.char.mem_op_loop test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharMemOpLoop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharMemOpLoop").encode("ascii"))

    def ibex_char_flash_read(self, flash_region=3) -> None:
        """Starts the ibex.char.flash_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharFlashRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharFlashRead").encode("ascii"))
        parameters = {"flash_region": flash_region}
        self.target.write(json.dumps(parameters).encode("ascii"))

    def ibex_char_flash_write(self, flash_region=3) -> None:
        """Starts the ibex.char.flash_write test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharFlashWrite command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharFlashWrite").encode("ascii"))
        parameters = {"flash_region": flash_region}
        self.target.write(json.dumps(parameters).encode("ascii"))

    def ibex_char_sram_read(self) -> None:
        """Starts the ibex.char.sram_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramRead").encode("ascii"))

    def ibex_char_sram_read_ret(self) -> None:
        """Starts the ibex.char.sram_read_ret test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramReadRet command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramReadRet").encode("ascii"))

    def ibex_char_sram_write_static_unrolled(self) -> None:
        """Starts the ibex.char.sram_write_static_unrolled test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramWriteStaticUnrolled command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharSramWriteStaticUnrolled").encode("ascii"))

    def ibex_char_sram_write_read(self) -> None:
        """Starts the ibex.char.sram_write_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramWriteRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramWriteRead").encode("ascii"))

    def ibex_char_sram_write_read_alt(self) -> None:
        """Starts the ibex.char.sram_write_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramWriteRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramWriteReadAlt").encode("ascii"))

    def ibex_char_sram_write(self) -> None:
        """Starts the ibex.char.sram_write test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramWrite command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramWrite").encode("ascii"))

    def ibex_char_sram_static(self) -> None:
        """Starts the ibex.char.sram_static test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSramStatic command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSramStatic").encode("ascii"))

    def ibex_char_single_beq(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSingleBeq command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSingleBeq").encode("ascii"))

    def ibex_char_addi_single_beq(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBeq command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBeq").encode("ascii"))

    def ibex_char_addi_single_beq_neg(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBeqNeg command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBeqNeg").encode("ascii"))

    def ibex_char_addi_single_beq_cm(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBeqCm command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBeqCm").encode("ascii"))

    def ibex_char_addi_single_beq_cm2(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBeqCm2 command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBeqCm2").encode("ascii"))

    def ibex_char_addi_single_bne(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBne command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBne").encode("ascii"))

    def ibex_char_single_bne(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharSingleBne command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharSingleBne").encode("ascii"))

    def ibex_char_addi_single_bne_neg(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharAddiSingleBneNeg command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharAddiSingleBneNeg").encode("ascii"))

    def ibex_char_conditional_branch_beq(self) -> None:
        """Starts the ibex.char.conditional_branch_beq test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBeq command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBeq").encode("ascii"))

    def ibex_char_conditional_branch_bne(self) -> None:
        """Starts the ibex.char.conditional_branch_bne test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBne command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBne").encode("ascii"))

    def ibex_char_conditional_branch_bge(self) -> None:
        """Starts the ibex.char.conditional_branch_bge test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBge command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBge").encode("ascii"))

    def ibex_char_conditional_branch_bgeu(self) -> None:
        """Starts the ibex.char.conditional_branch_bgeu test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBgeu command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBgeu").encode("ascii"))

    def ibex_char_conditional_branch_blt(self) -> None:
        """Starts the ibex.char.conditional_branch_blt test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBglt command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBlt").encode("ascii"))

    def ibex_char_conditional_branch_bltu(self) -> None:
        """Starts the ibex.char.conditional_branch_bltu test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCondBranchBgltu command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCondBranchBltu").encode("ascii"))

    def ibex_char_unconditional_branch(self) -> None:
        """Starts the ibex.char.unconditional_branch test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUncondBranch command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUncondBranch").encode("ascii"))

    def ibex_char_unconditional_branch_nop(self) -> None:
        """Starts the ibex.char.unconditional_branch_nop test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUncondBranchNop command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharUncondBranchNop").encode("ascii"))

    def ibex_char_hardened_check_eq_complement_branch(self) -> None:
        """Starts the ibex.char.hardened_check_eq_complement_branch test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheckComplementBranch command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharHardenedCheckComplementBranch").encode("ascii"))

    def ibex_char_register_file(self) -> None:
        """Starts the ibex.char.register_file test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharRegisterFile command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharRegisterFile").encode("ascii"))

    def ibex_char_register_file_read(self) -> None:
        """Starts the ibex.char.register_file_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharRegisterFileRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharRegisterFileRead").encode("ascii"))

    def init(
        self,
        core_config: dict = common_library.default_core_config,
        sensor_config: dict = common_library.default_sensor_config,
        alert_config: dict = common_library.default_alert_config,
    ) -> tuple:
        """Initialize the Ibex FI code on the chip.

        Returns:
            Device id
            The sensor control config
            The alert config
            The owner page
            The boot log
            The boot measurements
            The testOS version
        """

        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))

        # Write each configuration block to the target.
        self.target.write(json.dumps(core_config).encode("ascii"))
        self.target.write(json.dumps(sensor_config).encode("ascii"))
        self.target.write(json.dumps(alert_config).encode("ascii"))

        device_id = self.target.read_response()
        sensors = self.target.read_response()
        alerts = self.target.read_response()
        owner_page = self.target.read_response()
        boot_log = self.target.read_response()
        boot_measurements = self.target.read_response()
        version = self.target.read_response()
        return (
            device_id,
            sensors,
            alerts,
            owner_page,
            boot_log,
            boot_measurements,
            version,
        )

    def start_test(self, cfg: dict) -> None:
        """Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            cfg: Config dict containing the selected test.
        """
        test_function = getattr(self, cfg["test"]["which_test"])
        test_function()

    def ibex_char_csr_write(self) -> None:
        """Starts the ibex.fi.char.csr_write test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCsrWrite command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCsrWrite").encode("ascii"))

    def ibex_char_csr_read(self) -> None:
        """Starts the ibex.fi.char.csr_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCsrRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCsrRead").encode("ascii"))

    def ibex_char_csr_combi(self, trigger, ref_values) -> None:
        """Starts the ibex.fi.char.csr_combi test.

        Args:
            trigger: Which triggers to raise.
            ref_values: The values to be written in the CSRs.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCsrCombi command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCsrCombi").encode("ascii"))
        data = {"trigger": trigger, "ref_values": ref_values}
        self.target.write(json.dumps(data).encode("ascii"))

    def ibex_address_translation_config(self) -> None:
        """Starts the ibex.fi.address_translation_config test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # AddressTranslationCfg command.
        time.sleep(0.01)
        self.target.write(json.dumps("AddressTranslationCfg").encode("ascii"))

    def ibex_address_translation(self) -> None:
        """Starts the ibex.fi.address_translation test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # AddressTranslation command.
        time.sleep(0.01)
        self.target.write(json.dumps("AddressTranslation").encode("ascii"))

    def ibex_char_hardened_check_eq_unimp(self) -> None:
        """Starts the ibex.fi.char.hardened_check_eq_unimp test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheckUnimp command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharHardenedCheckUnimp").encode("ascii"))

    def ibex_char_hardened_check_eq_2_unimps(self) -> None:
        """Starts the ibex.fi.char.hardened_check_eq_2_unimps test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheck2Unimps command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharHardenedCheck2Unimps").encode("ascii"))

    def ibex_char_hardened_check_eq_3_unimps(self) -> None:
        """Starts the ibex.fi.char.hardened_check_eq_3_unimps test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheck3Unimps command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharHardenedCheck3Unimps").encode("ascii"))

    def ibex_char_hardened_check_eq_4_unimps(self) -> None:
        """Starts the ibex.fi.char.hardened_check_eq_4_unimps test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheck4Unimps command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharHardenedCheck4Unimps").encode("ascii"))

    def ibex_char_hardened_check_eq_5_unimps(self) -> None:
        """Starts the ibex.fi.char.hardened_check_eq_5_unimps test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheck5Unimps command.
        time.sleep(0.01)
        self.target.write(
            json.dumps("CharHardenedCheck5Unimps").encode("ascii"))

    def ibex_char_combi(self) -> None:
        """Starts the ibex.fi.char.combi test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharCombi command.
        time.sleep(0.01)
        self.target.write(json.dumps("CharCombi").encode("ascii"))

    def ibex_char_otp_data_read(self) -> None:
        """Starts the ibex.fi.otp.data_read test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # OtpDataRead command.
        time.sleep(0.01)
        self.target.write(json.dumps("OtpDataRead").encode("ascii"))

    def ibex_char_otp_read_lock(self) -> None:
        """Starts the ibex.fi.otp.read_lock test."""
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # OtpReadLock command.
        time.sleep(0.01)
        self.target.write(json.dumps("OtpReadLock").encode("ascii"))

    def ibex_characterise(self) -> None:
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharHardenedCheck3Unimps command.
        time.sleep(0.01)
        self.target.write(json.dumps("Characterisation").encode("ascii"))
