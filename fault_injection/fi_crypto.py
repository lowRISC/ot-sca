#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import struct
from datetime import datetime
from pathlib import Path

import yaml
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256, SHA384, SHA512
from fi_gear.fi_gear import FIGear
from tqdm import tqdm

import util.helpers as helpers
from fault_injection.project_library.project import (FIProject, FISuccess,
                                                     ProjectConfig)
from target.communication.fi_crypto_commands import OTFICrypto
from target.targets import Target, TargetConfig
from util import plot

logger = logging.getLogger()


def words_to_bytes(dword_array):
    if not isinstance(dword_array, list):
        raise TypeError("Input must be a list of 32-bit integers.")

    byte_list = bytearray()
    for dword in dword_array:
        if not isinstance(dword, int) or not (0 <= dword <= 0xFFFFFFFF):
            raise ValueError(
                "Each element in the array must be a 32-bit integer (0 to 0xFFFFFFFF)."
            )
        byte_list.extend(struct.pack(">I", dword))
    return bytes(byte_list)


def bytes_to_words(byte_array):
    if not isinstance(byte_array, (bytes, bytearray)):
        raise TypeError("Input must be a bytes object or bytearray.")

    word_list = []
    for i in range(0, len(byte_array), 4):
        chunk = byte_array[i:i + 4]
        word = struct.unpack(">I", chunk)[0]
        word_list.append(word)
    return word_list


def setup(cfg: dict, project: Path):
    """Setup target, FI gear, and project.

    Args:
        cfg: The configuration for the current experiment.
        project: The path for the project file.

    Returns:
        The target, FI gear, and project.
    """
    # Calculate pll_frequency of the target.
    # target_freq = pll_frequency * target_clk_mult
    # target_clk_mult is a hardcoded constant in the FPGA bitstream.
    cfg["target"]["pll_frequency"] = (cfg["target"]["target_freq"] /
                                      cfg["target"]["target_clk_mult"])

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type=cfg["target"]["target_type"],
        fw_bin=cfg["target"]["fw_bin"],
        pll_frequency=cfg["target"]["pll_frequency"],
        bitstream=cfg["target"].get("fpga_bitstream"),
        force_program_bitstream=cfg["target"].get("force_program_bitstream"),
        baudrate=cfg["target"].get("baudrate"),
        port=cfg["target"].get("port"),
        usb_serial=cfg["target"].get("usb_serial"),
        interface=cfg["target"].get("interface"),
        husky_serial=cfg["fisetup"].get("usb_serial"),
        opentitantool=cfg["target"]["opentitantool"],
    )
    target = Target(target_cfg)

    # Init FI gear.
    fi_gear = FIGear(cfg)

    # Init project.
    project_cfg = ProjectConfig(
        type=cfg["fiproject"]["project_db"],
        path=project,
        overwrite=True,
        fi_threshold=cfg["fiproject"].get("project_mem_threshold"),
    )
    project = FIProject(project_cfg)
    project.create_project()

    return target, fi_gear, project


def print_fi_statistic(fi_results: list) -> None:
    """Print FI Statistic.

    Prints the number of FISuccess.SUCCESS, FISuccess.EXPRESPONSE, and
    FISuccess.NORESPONSE.

    Args:
        fi_results: The FI results.
    """
    num_total = len(fi_results)
    num_succ = round((fi_results.count(FISuccess.SUCCESS) / num_total) * 100,
                     2)
    num_exp = round(
        (fi_results.count(FISuccess.EXPRESPONSE) / num_total) * 100, 2)
    num_no = round((fi_results.count(FISuccess.NORESPONSE) / num_total) * 100,
                   2)
    logger.info(
        f"{num_total} faults, {fi_results.count(FISuccess.SUCCESS)}"
        f"({num_succ}%) successful, {fi_results.count(FISuccess.EXPRESPONSE)}"
        f"({num_exp}%) expected, and {fi_results.count(FISuccess.NORESPONSE)}"
        f"({num_no}%) no response.")


def fi_parameter_sweep(cfg: dict, target: Target, fi_gear, project: FIProject,
                       ot_communication: OTFICrypto) -> None:
    """Fault parameter sweep.

    Sweep through the fault parameter space.

    Args:
        cfg: The FI project configuration.
        target: The OpenTitan target.
        fi_gear: The FI gear to use.
        project: The project to store the results.
        ot_communication: The OpenTitan Crypto FI communication interface.
    Returns:
        device_id: The ID of the target device.
        sensors: The sensor info.
        alerts: The alert info.
        owner_page: The owner info page.
        boot_log: The boot log.
        boot_measurments: The boot measurements.
        version: The testOS version.
    """
    # Configure the Crypto FI code on the target.
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        ot_communication.init(
            cfg["test"]["core_config"],
            cfg["test"]["sensor_config"],
            cfg["test"]["alert_config"],
        ))
    # Store results in array for a quick access.
    fi_results = []
    # Start the parameter sweep.
    remaining_iterations = fi_gear.get_num_fault_injections()
    with tqdm(total=remaining_iterations,
              desc="Injecting",
              ncols=80,
              unit=" different faults") as pbar:
        while remaining_iterations > 0:
            # Get fault parameters (e.g., trigger delay, glitch voltage).
            fault_parameters = fi_gear.generate_fi_parameters()

            # Arm the FI gear.
            fi_gear.arm_trigger(fault_parameters)

            # Get the input arguments.
            args = {}
            if "aes" in cfg["test"]["which_test"]:
                args["plaintext"] = cfg["test"]["plaintext"]
                args["key"] = cfg["test"]["key"]
            elif "hmac" in cfg["test"]["which_test"]:
                args["msg"] = cfg["test"]["msg"]
                args["key"] = cfg["test"]["key"]
                args["trigger"] = cfg["test"]["trigger"]
                args["enable_hmac"] = cfg["test"]["enable_hmac"]
                args["hash_mode"] = cfg["test"]["hash_mode"]
                # These are hard coded for now.
                args["message_endianness_big"] = False
                args["digest_endianness_big"] = False
                args["key_endianness_big"] = False

            # Start test on OpenTitan.
            ot_communication.start_test(cfg, **args)

            # Read response.
            response = target.read_response()
            response_compare = response

            if "aes" in cfg["test"]["which_test"]:
                cipher_gen = AES.new(bytes(cfg["test"]["key"]), AES.MODE_ECB)
                expected_response = [
                    x for x in cipher_gen.encrypt(
                        bytes(cfg["test"]["plaintext"]))
                ]
                exp_json = {
                    "ciphertext": expected_response,
                    "err_status": 0,
                    "alerts": [0, 0, 0],
                    "ast_alerts": [0, 0],
                }
                expected_response = json.dumps(exp_json, separators=(",", ":"))
            elif "hmac" in cfg["test"]["which_test"]:
                if not cfg["test"]["enable_hmac"]:
                    if cfg["test"]["hash_mode"] == 0:
                        sha = SHA256.new()
                    elif cfg["test"]["hash_mode"] == 1:
                        sha = SHA384.new()
                    elif cfg["test"]["hash_mode"] == 2:
                        sha = SHA512.new()
                    else:
                        logger.info("Error: Hash mode not recognized.")
                        return
                    sha.update(bytes(cfg["test"]["msg"]))
                    expected_response = bytes_to_words(bytearray(sha.digest()))
                    expected_response.reverse()
                    if cfg["test"]["hash_mode"] == 0:
                        expected_response += [0] * 8
                    elif cfg["test"]["hash_mode"] == 1:
                        expected_response += [0] * 4
                else:
                    if cfg["test"]["hash_mode"] == 0:
                        hmac = HMAC.new(
                            key=bytes(words_to_bytes(cfg["test"]["key"])),
                            digestmod=SHA256,
                        )
                    elif cfg["test"]["hash_mode"] == 1:
                        hmac = HMAC.new(
                            key=bytes(words_to_bytes(cfg["test"]["key"])),
                            digestmod=SHA384,
                        )
                    elif cfg["test"]["hash_mode"] == 2:
                        hmac = HMAC.new(
                            key=bytes(words_to_bytes(cfg["test"]["key"])),
                            digestmod=SHA512,
                        )
                    else:
                        logger.info("Error: Hash mode not recognized.")
                        return
                    hmac.update(bytes(cfg["test"]["msg"]))
                    expected_response = bytes_to_words(bytearray(
                        hmac.digest()))
                    expected_response.reverse()
                    if cfg["test"]["hash_mode"] == 0:
                        expected_response += [0] * 8
                    elif cfg["test"]["hash_mode"] == 1:
                        expected_response += [0] * 4
                expected_response = json.dumps(exp_json, separators=(",", ":"))
            elif "kmac" in cfg["test"]["which_test"]:
                expected_response = '{"digest":[184,34,91,108,231,47,251,27], \
                    "err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'

                exp_json = json.loads(expected_response)
            elif "shadow_reg_access" in cfg["test"]["which_test"]:
                expected_response = (
                    '{"result":[68162304,0,0],"err_status":0,"ast_alerts":[0,0]}'
                )
                exp_json = json.loads(expected_response)

            # Compare response. If no response is received, the device mostly
            # crashed or was resetted.
            if response_compare == "":
                # No UART response received.
                fi_result = FISuccess.NORESPONSE
                # Resetting OT as it most likely crashed.
                ot_communication = target.reset_target(com_reset=True)
                # Re-establish UART connection.
                ot_communication = OTFICrypto(target)
                # Configure the Crypto FI code on the target.
                ot_communication.init(
                    cfg["test"]["core_config"],
                    cfg["test"]["sensor_config"],
                    cfg["test"]["alert_config"],
                )
                # Reset FIGear if necessary.
                fi_gear.reset()
            else:
                # If the test decides to ignore alerts triggered by the alert
                # handler, remove it from the received and expected response.
                # In the database, the received alert is still available for
                # further diagnosis.
                if cfg["test"]["ignore_alerts"]:
                    resp_json = json.loads(response_compare)
                    if "alerts" in resp_json:
                        del resp_json["alerts"]
                        response_compare = json.dumps(resp_json,
                                                      separators=(",", ":"))
                    if "alerts" in exp_json:
                        del exp_json["alerts"]
                        expected_response = json.dumps(exp_json,
                                                       separators=(",", ":"))

                # Check if result is expected result (FI failed) or unexpected
                # result (FI successful).
                fi_result = FISuccess.SUCCESS
                if response_compare == expected_response:
                    # Expected result received. No FI effect.
                    fi_result = FISuccess.EXPRESPONSE

            # Store result into FIProject.
            project.append_firesult(
                response=response,
                fi_result=fi_result,
                trigger_delay=fault_parameters.get("trigger_delay"),
                glitch_voltage=fault_parameters.get("glitch_voltage"),
                glitch_width=fault_parameters.get("glitch_width"),
                x_pos=fault_parameters.get("x_pos"),
                y_pos=fault_parameters.get("y_pos"),
            )
            fi_results.append(fi_result)

            remaining_iterations -= 1
            pbar.update(1)
    print_fi_statistic(fi_results)
    return device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version


def print_plot(project: FIProject, config: dict, file: Path) -> None:
    """Print plot of traces.

    Printing the plot helps to narrow down the fault injection parameters.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The file path.
    """
    if config["fiproject"]["show_plot"]:
        plot.save_fi_plot_to_file(config, project, file)
        logger.info("Created plot.")
        logger.info(f"Created plot: "
                    f'{Path(str(file) + ".html").resolve()}')


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Parse the provided arguments.
    args = helpers.parse_arguments(argv)

    # Load configuration from file.
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Setup the target, FI gear, and project.
    target, fi_gear, project = setup(cfg, args.project)

    # Establish communication interface with OpenTitan.
    ot_communication = OTFICrypto(target)

    # FI parameter sweep.
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        fi_parameter_sweep(cfg, target, fi_gear, project, ot_communication))

    # Print plot.
    print_plot(
        project.get_firesults(start=0, end=cfg["fiproject"]["num_plots"]),
        cfg,
        args.project,
    )

    # Save metadata.
    metadata = {}
    metadata["device_id"] = device_id
    metadata["sensors"] = sensors
    metadata["alerts"] = alerts
    metadata["owner_page"] = owner_page
    metadata["boot_log"] = boot_log
    metadata["boot_measurements"] = boot_measurements
    metadata["version"] = version
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    # Store bitstream information.
    metadata["fpga_bitstream_path"] = cfg["target"].get("fpga_bitstream")
    if cfg["target"].get("fpga_bitstream") is not None:
        metadata["fpga_bitstream_crc"] = helpers.file_crc(
            cfg["target"]["fpga_bitstream"])
    if args.save_bitstream:
        metadata["fpga_bitstream"] = helpers.get_binary_blob(
            cfg["target"]["fpga_bitstream"])
    # Store binary information.
    metadata["fw_bin_path"] = cfg["target"]["fw_bin"]
    metadata["fw_bin_crc"] = helpers.file_crc(cfg["target"]["fw_bin"])
    if args.save_binary:
        metadata["fw_bin"] = helpers.get_binary_blob(cfg["target"]["fw_bin"])
    # Store user provided notes.
    metadata["notes"] = args.notes
    # Store the Git hash.
    metadata["git_hash"] = helpers.get_git_hash()
    # Write metadata into project database.
    project.write_metadata(metadata)

    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
