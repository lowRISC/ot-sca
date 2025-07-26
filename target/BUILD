# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party/python:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "chip",
    srcs = ["chip.py"],
)

py_library(
    name = "dut",
    srcs = ["dut.py"],
    deps = [requirement("pyserial")],
)

py_library(
    name = "data_generator",
    srcs = ["data_generator.py"],
    deps = [requirement("pycryptodome")],
)
