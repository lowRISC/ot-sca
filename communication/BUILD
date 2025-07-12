package(default_visibility = ["//visibility:public"])

load("@//third_party/python:requirements.bzl", "requirement")

py_library(
    name = "chip",
    srcs = ["chip.py"],
)

py_library(
    name = "dut",
    srcs = ["dut.py"],
    deps = [requirement("pyserial"),],
)

py_library(
    name = "fi_ibex_commands",
    srcs = ["fi_ibex_commands.py"],
)