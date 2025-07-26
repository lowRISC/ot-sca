# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party/python:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "fi_ibex_test",
    testonly = True,
    srcs = ["test_scripts/fi_ibex_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_ibex_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_ibex_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_ibex_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_ibex_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "fi_crypto_test",
    testonly = True,
    srcs = ["test_scripts/fi_crypto_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_crypto_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_crypto_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
)

py_test(
    name = "fi_otbn_test",
    testonly = True,
    srcs = ["test_scripts/fi_otbn_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_otbn_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_otbn_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_otbn_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_otbn_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "fi_otp_test",
    testonly = True,
    srcs = ["test_scripts/fi_otp_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_otp_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_otp_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "fi_rom_test",
    testonly = True,
    srcs = ["test_scripts/fi_rom_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_rom_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_rom_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "fi_rng_test",
    testonly = True,
    srcs = ["test_scripts/fi_rng_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_fi_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fisilicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_fi_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":fi_rng_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_rng_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "fi_ibex_functions",
    srcs = ["host_scripts/fi_ibex_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_ibex_commands",
    ],
)

py_library(
    name = "fi_crypto_functions",
    srcs = ["host_scripts/fi_crypto_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_crypto_commands",
    ],
)

py_library(
    name = "fi_otbn_functions",
    srcs = ["host_scripts/fi_otbn_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_otbn_commands",
    ],
)

py_library(
    name = "fi_otp_functions",
    srcs = ["host_scripts/fi_otp_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_otp_commands",
    ],
)

py_library(
    name = "fi_rom_functions",
    srcs = ["host_scripts/fi_rom_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_rom_commands",
    ],
)

py_library(
    name = "fi_rng_functions",
    srcs = ["host_scripts/fi_rng_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:fi_rng_commands",
    ],
)
