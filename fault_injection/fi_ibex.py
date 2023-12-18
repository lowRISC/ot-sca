#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from fi_gear.fi_gear import FIGear
from project_library.project import FIProject, FISuccess, ProjectConfig
from tqdm import tqdm

sys.path.append("../")
import util.helpers as helpers  # noqa: E402
from target.communication.fi_ibex_commands import OTUART  # noqa: E402
from target.communication.fi_ibex_commands import OTFIIbex  # noqa: E402
from target.cw_fpga import CWFPGA  # noqa: E402
from util import plot  # noqa: E402

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

    # Init target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target = CWFPGA(
        bitstream = cfg["target"]["fpga_bitstream"],
        force_programming = cfg["target"]["force_program_bitstream"],
        firmware = cfg["target"]["fw_bin"],
        pll_frequency = cfg["target"]["pll_frequency"],
        baudrate = cfg["target"]["baudrate"],
        output_len = cfg["target"]["output_len_bytes"],
        protocol = cfg["target"]["protocol"]
    )

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


def fi_parameter_sweep(cfg: dict, cwtarget: CWFPGA, fi_gear,
                       project: FIProject, ot_communication: OTFIIbex) -> None:
    """ Fault parameter sweep.

    Sweep through the fault parameter space.

    Args:
        cfg: The FI project configuration.
        cwtarget: The OpenTitan target.
        fi_gear: The FI gear to use.
        project: The project to store the results.
        ot_communication: The OpenTitan Ibex FI communication interface.
    """
    # Configure the trigger.
    ot_communication.init_trigger()
    # Start the parameter sweep.
    remaining_iterations = cfg["fisetup"]["num_iterations"]
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

            # Compare response.
            # Check if result is expected result (FI failed), unexpected result
            # (FI successful), or no response (FI failed.)
            fi_result = FISuccess.SUCCESS
            if response == cfg["test"]["expected_result"]:
                # Expected result received. No FI effect.
                fi_result = FISuccess.EXPRESPONSE
            elif response == "":
                # No UART response received.
                fi_result = FISuccess.NORESPONSE
                # Resetting OT as it most likely crashed.
                cwtarget.reset_target()
                # Re-establish UART connection.
                ot_uart = OTUART(protocol = cfg["target"]["protocol"],
                                 port = cfg["target"]["port"])
                ot_communication = OTFIIbex(ot_uart.uart)
                # Configure the trigger.
                ot_communication.init_trigger()
                # Reset FIGear if necessary.
                fi_gear.reset()

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

            remaining_iterations -= 1
            pbar.update(1)


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
    ot_uart = OTUART(protocol = cfg["target"]["protocol"],
                     port = cfg["target"]["port"])
    ot_communication = OTFIIbex(ot_uart.uart)

    # FI parameter sweep.
    fi_parameter_sweep(cfg, target, fi_gear, project, ot_communication)

    # Print plot.
    print_plot(project.get_firesults(start=0, end=cfg["fiproject"]["num_plots"]),
               cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata["cfg"] = cfg
    metadata["fpga_bitstream"] = cfg["target"]["fpga_bitstream"]
    # TODO: Store binary into database instead of binary path.
    # (Issue lowrisc/ot-sca#214)
    metadata["fw_bin"] = cfg["target"]["fw_bin"]
    metadata["notes"] = args.notes
    project.write_metadata(metadata)

    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
