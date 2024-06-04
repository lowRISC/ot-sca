# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Ibex FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
from target.communication.otfi import OTFI
from target.communication.otfi_test import OTFITest


class OTFIIbex(OTFI):
    TESTS = [
        OTFITest("char_unrolled_reg_op_loop"),
        OTFITest("char_unrolled_mem_op_loop"),
        OTFITest("char_reg_op_loop"),
        OTFITest("char_mem_op_loop"),
        OTFITest("char_flash_read"),
        OTFITest("char_flash_write"),
        OTFITest("char_sram_read"),
        OTFITest("char_sram_write_static_unrolled"),
        OTFITest("char_sram_write_read"),
        OTFITest("char_sram_write"),
        OTFITest("char_sram_static"),
        OTFITest("char_conditional_branch_beq", "CharCondBranchBeq"),
        OTFITest("char_conditional_branch_bne", "CharCondBranchBne"),
        OTFITest("char_conditional_branch_bge", "CharCondBranchBge"),
        OTFITest("char_conditional_branch_bgeu", "CharCondBranchBgeu"),
        OTFITest("char_conditional_branch_blt", "CharCondBranchBlt"),
        OTFITest("char_conditional_branch_bltu", "CharCondBranchBltu"),
        OTFITest("char_unconditional_branch", "CharUncondBranch"),
        OTFITest("char_unconditional_branch_nop", "CharUncondBranchNop"),
        OTFITest("char_register_file"),
        OTFITest("char_register_file_read"),
        OTFITest("char_csr_write"),
        OTFITest("char_csr_read"),
        OTFITest("address_translation"),
        OTFITest("address_translation_config", "AddressTranslationCfg"),
    ]

    def __init__(self, target) -> None:
        super().__init__(target, "Ibex")
