package(default_visibility = ["//visibility:public"])

load("@//third_party/python:requirements.bzl", "requirement")

py_binary(
    name = "fi_ibex_test",
    srcs = ["test_scripts/fi_ibex_test.py"],
    testonly = True,
    deps = [
        ":fi_ibex_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_ibex_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "fi_crypto_test",
    srcs = ["test_scripts/fi_crypto_test.py"],
    testonly = True,
    deps = [
        ":fi_crypto_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_crypto_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "fi_otbn_test",
    srcs = ["test_scripts/fi_otbn_test.py"],
    testonly = True,
    deps = [
        ":fi_otbn_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_otbn_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "fi_otp_test",
    srcs = ["test_scripts/fi_otp_test.py"],
    testonly = True,
    deps = [
        ":fi_otp_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_otp_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_library(
    name = "fi_ibex_functions",
    srcs = ["host_scripts/fi_ibex_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_ibex_commands",
    ],
)

py_library(
    name = "fi_crypto_functions",
    srcs = ["host_scripts/fi_crypto_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_crypto_commands",
    ],
)

py_library(
    name = "fi_otbn_functions",
    srcs = ["host_scripts/fi_otbn_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_otbn_commands",
    ],
)

py_library(
    name = "fi_otp_functions",
    srcs = ["host_scripts/fi_otp_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_otp_commands",
    ],
)