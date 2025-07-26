# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party/python:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "sca_ibex_test",
    testonly = True,
    srcs = ["test_scripts/sca_ibex_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_ibex_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_ibex_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "sca_aes_test",
    testonly = True,
    srcs = ["test_scripts/sca_aes_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_aes_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_aes_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
)

py_test(
    name = "sca_hmac_test",
    testonly = True,
    srcs = ["test_scripts/sca_hmac_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_hmac_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_hmac_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
)

py_test(
    name = "sca_kmac_test",
    testonly = True,
    srcs = ["test_scripts/sca_kmac_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_kmac_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_kmac_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
)

py_test(
    name = "sca_otbn_test",
    testonly = True,
    srcs = ["test_scripts/sca_otbn_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_otbn_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_otbn_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "sca_sha3_test",
    testonly = True,
    srcs = ["test_scripts/sca_sha3_test.py"],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool",
    ] + select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_rom_with_fake_keys",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_sival_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_fpga_cw310_test_rom",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_a2_rom_ext",
        ],
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_sival_rom_ext",
        ],
        "//conditions:default": [
            "@lowrisc_opentitan//sw/device/tests/penetrationtests/firmware:pen_test_sca_silicon_owner_gb_rom_ext",
        ],
    }),
    env = select({
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_gb_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_rom_with_fake_keys": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_rom_with_fake_keys",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_sival_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_fpga_cw310_test_rom": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_fpga_cw310_test_rom",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_a2_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_a2_rom_ext",
        },
        "@lowrisc_opentitan//sw/device/tests/penetrationtests/config:env_silicon_owner_sival_rom_ext": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_sival_rom_ext",
        },
        "//conditions:default": {
            "SELECTED_FIRMWARE_TARGET": "pen_test_sca_silicon_owner_gb_rom_ext",
        },
    }),
    deps = [
        ":sca_sha3_functions",
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_sha3_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
)

py_library(
    name = "sca_ibex_functions",
    srcs = ["host_scripts/sca_ibex_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_ibex_commands",
        "//target/communication:sca_prng_commands",
    ],
)

py_library(
    name = "sca_aes_functions",
    srcs = ["host_scripts/sca_aes_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_aes_commands",
        "//target/communication:sca_prng_commands",
        "//target/communication:sca_trigger_commands",
    ],
)

py_library(
    name = "sca_hmac_functions",
    srcs = ["host_scripts/sca_hmac_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_hmac_commands",
        "//target/communication:sca_prng_commands",
    ],
)

py_library(
    name = "sca_kmac_functions",
    srcs = ["host_scripts/sca_kmac_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_kmac_commands",
        "//target/communication:sca_prng_commands",
        "//target/communication:sca_trigger_commands",
    ],
)

py_library(
    name = "sca_otbn_functions",
    srcs = ["host_scripts/sca_otbn_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_otbn_commands",
    ],
)

py_library(
    name = "sca_sha3_functions",
    srcs = ["host_scripts/sca_sha3_functions.py"],
    deps = [
        "//target:chip",
        "//target:dut",
        "//target/communication:sca_prng_commands",
        "//target/communication:sca_sha3_commands",
        "//target/communication:sca_trigger_commands",
    ],
)
