package(default_visibility = ["//visibility:public"])

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

py_library(
    name = "fi_ibex_functions",
    srcs = ["host_scripts/fi_ibex_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:fi_ibex_commands",
    ],
)