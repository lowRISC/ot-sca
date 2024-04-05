#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from fi_gear.fi_gear import FIGear
from project_library.project import FIProject, FISuccess, ProjectConfig
from tqdm import tqdm

import util.helpers as helpers
from target.communication.fi_ibex_commands import OTFIIbex
from target.targets import Target, TargetConfig
from util import plot

logger = logging.getLogger()


def setup(cfg: dict, project: Path):
    """ Setup target, FI gear, and project.

    Args:
        cfg: The configuration for the current experiment.
        project: The path for the project file.

    Returns:
        The target, FI gear, and project.
    """
    # Calculate pll_frequency of the target.
    # target_freq = pll_frequency * target_clk_mult
    # target_clk_mult is a hardcoded constant in the FPGA bitstream.
    cfg["target"]["pll_frequency"] = cfg["target"]["target_freq"] / cfg["target"]["target_clk_mult"]

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type = cfg["target"]["target_type"],
        fw_bin = cfg["target"]["fw_bin"],
        protocol = cfg["target"]["protocol"],
        pll_frequency = cfg["target"]["pll_frequency"],
        bitstream = cfg["target"].get("fpga_bitstream"),
        force_program_bitstream = cfg["target"].get("force_program_bitstream"),
        baudrate = cfg["target"].get("baudrate"),
        port = cfg["target"].get("port"),
        output_len = cfg["target"].get("output_len_bytes"),
        usb_serial = cfg["target"].get("usb_serial")
    )
    target = Target(target_cfg)

    # Init FI gear.
    fi_gear = FIGear(cfg)

    # Init project.
    project_cfg = ProjectConfig(type = cfg["fiproject"]["project_db"],
                                path = project,
                                overwrite = True,
                                fi_threshold = cfg["fiproject"].get("project_mem_threshold")
                                )
    project = FIProject(project_cfg)
    project.create_project()

    return target, fi_gear, project


def print_fi_statistic(fi_results: list) -> None:
    """ Print FI Statistic.

    Prints the number of FISuccess.SUCCESS, FISuccess.EXPRESPONSE, and
    FISuccess.NORESPONSE.

    Args:
        fi_results: The FI results.
    """
    num_total = len(fi_results)
    num_succ = round((fi_results.count(FISuccess.SUCCESS) / num_total) * 100, 2)
    num_exp = round((fi_results.count(FISuccess.EXPRESPONSE) / num_total) * 100, 2)
    num_no = round((fi_results.count(FISuccess.NORESPONSE) / num_total) * 100, 2)
    logger.info(f"{num_total} faults, {fi_results.count(FISuccess.SUCCESS)}"
                f"({num_succ}%) successful, {fi_results.count(FISuccess.EXPRESPONSE)}"
                f"({num_exp}%) expected, and {fi_results.count(FISuccess.NORESPONSE)}"
                f"({num_no}%) no response.")


def fi_parameter_sweep(cfg: dict, target: Target, fi_gear,
                       project: FIProject, ot_communication: OTFIIbex) -> None:
    """ Fault parameter sweep.

    Sweep through the fault parameter space.

    Args:
        cfg: The FI project configuration.
        target: The OpenTitan target.
        fi_gear: The FI gear to use.
        project: The project to store the results.
        ot_communication: The OpenTitan Ibex FI communication interface.
    """
    # Configure the Ibex FI code on the target.
    ot_communication.init()
    # Store results in array for a quick access.
    fi_results = []
    # Start the parameter sweep.
    remaining_iterations = fi_gear.get_num_fault_injections()
    with tqdm(total=remaining_iterations, desc="Injecting", ncols=80,
              unit=" different faults") as pbar:
        while remaining_iterations > 0:
            # Get fault parameters (e.g., trigger delay, glitch voltage).
            fault_parameters = fi_gear.generate_fi_parameters()

            # Arm the FI gear.
            fi_gear.arm_trigger(fault_parameters)

            # Start test on OpenTitan.
            ot_communication.start_test(cfg)

            # Read response.
            response = ot_communication.read_response()
            response_compare = response
            expected_response = cfg["test"]["expected_result"]

            # Compare response.
            if response_compare == "":
                # No UART response received.
                fi_result = FISuccess.NORESPONSE
                # Resetting OT as it most likely crashed.
                ot_communication = target.reset_target(com_reset = True)
                # Re-establish UART connection.
                ot_communication = OTFIIbex(target)
                # Configure the Ibex FI code on the target.
                ot_communication.init()
                # Reset FIGear if necessary.
                fi_gear.reset()
            else:
                # If the test decides to ignore alerts triggered by the alert
                # handler, remove it from the received and expected response.
                # In the database, the received alert is still available for
                # further diagnosis.
                if cfg["test"]["ignore_alerts"]:
                    resp_json = json.loads(response_compare)
                    exp_json = json.loads(expected_response)
                    if "alerts" in resp_json:
                        del resp_json["alerts"]
                        response_compare = json.dumps(resp_json,
                                                      separators=(',', ':'))
                    if "alerts" in exp_json:
                        del exp_json["alerts"]
                        expected_response = json.dumps(exp_json,
                                                       separators=(',', ':'))

                # Check if result is expected result (FI failed), unexpected result
                # (FI successful), or no response (FI failed.)
                fi_result = FISuccess.SUCCESS
                if response_compare == expected_response:
                    # Expected result received. No FI effect.
                    fi_result = FISuccess.EXPRESPONSE

            # Store result into FIProject.
            project.append_firesult(
                response = response,
                fi_result = fi_result,
                trigger_delay = fault_parameters.get("trigger_delay"),
                glitch_voltage = fault_parameters.get("glitch_voltage"),
                glitch_width = fault_parameters.get("glitch_width"),
                x_pos = fault_parameters.get("x_pos"),
                y_pos = fault_parameters.get("y_pos")
            )
            fi_results.append(fi_result)

            remaining_iterations -= 1
            pbar.update(1)
    print_fi_statistic(fi_results)


def print_plot(project: FIProject, config: dict, file: Path) -> None:
    """ Print plot of traces.

    Printing the plot helps to narrow down the fault injection parameters.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The file path.
    """
    if config["fiproject"]["show_plot"]:
        plot.save_fi_plot_to_file(config, project, file)
        logger.info("Created plot.")
        logger.info(f'Created plot: '
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
    ot_communication = OTFIIbex(target)

    # FI parameter sweep.
    fi_parameter_sweep(cfg, target, fi_gear, project, ot_communication)

    # Print plot.
    print_plot(project.get_firesults(start=0, end=cfg["fiproject"]["num_plots"]),
               cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    # Store bitstream information.
    metadata["fpga_bitstream_path"] = cfg["target"].get("fpga_bitstream")
    if cfg["target"].get("fpga_bitstream") is not None:
        metadata["fpga_bitstream_crc"] = helpers.file_crc(cfg["target"]["fpga_bitstream"])
    if args.save_bitstream:
        metadata["fpga_bitstream"] = helpers.get_binary_blob(cfg["target"]["fpga_bitstream"])
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
