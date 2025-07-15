package(default_visibility = ["//visibility:public"])

load("@//third_party/python:requirements.bzl", "requirement")

py_binary(
    name = "sca_ibex_test",
    srcs = ["test_scripts/sca_ibex_test.py"],
    testonly = True,
    deps = [
        ":sca_ibex_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_ibex_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "sca_aes_test",
    srcs = ["test_scripts/sca_aes_test.py"],
    testonly = True,
    deps = [
        ":sca_aes_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_aes_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "sca_hmac_test",
    srcs = ["test_scripts/sca_hmac_test.py"],
    testonly = True,
    deps = [
        ":sca_hmac_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_hmac_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "sca_kmac_test",
    srcs = ["test_scripts/sca_kmac_test.py"],
    testonly = True,
    deps = [
        ":sca_kmac_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_kmac_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "sca_otbn_test",
    srcs = ["test_scripts/sca_otbn_test.py"],
    testonly = True,
    deps = [
        ":sca_otbn_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_otbn_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_binary(
    name = "sca_sha3_test",
    srcs = ["test_scripts/sca_sha3_test.py"],
    testonly = True,
    deps = [
        ":sca_sha3_functions",
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_sha3_commands",
        "//test/penetrationtests/util:utils",
        "@rules_python//python/runfiles",
        requirement("pycryptodome"),
    ],
    data = [
        "@lowrisc_opentitan//sw/host/opentitantool", 
    ] +
        select({
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
)

py_library(
    name = "sca_ibex_functions",
    srcs = ["host_scripts/sca_ibex_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_ibex_commands",
        "//communication:sca_prng_commands",
    ],
)

py_library(
    name = "sca_aes_functions",
    srcs = ["host_scripts/sca_aes_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_aes_commands",
        "//communication:sca_prng_commands",
        "//communication:sca_trigger_commands",
        "//communication:data_generator",
    ],
)

py_library(
    name = "sca_hmac_functions",
    srcs = ["host_scripts/sca_hmac_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_hmac_commands",
        "//communication:sca_prng_commands",
        "//communication:sca_trigger_commands",
    ],
)

py_library(
    name = "sca_kmac_functions",
    srcs = ["host_scripts/sca_kmac_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_kmac_commands",
        "//communication:sca_prng_commands",
        "//communication:sca_trigger_commands",
    ],
)

py_library(
    name = "sca_otbn_functions",
    srcs = ["host_scripts/sca_otbn_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_otbn_commands",
    ],
)

py_library(
    name = "sca_sha3_functions",
    srcs = ["host_scripts/sca_sha3_functions.py"],
    deps = [
        "//communication:dut",
        "//communication:chip",
        "//communication:sca_prng_commands",
        "//communication:sca_trigger_commands",
        "//communication:sca_sha3_commands",
    ],
)