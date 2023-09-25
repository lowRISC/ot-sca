# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Support for capturing traces using LeCroy WaveRunner 9104."""

import re

import numpy as np
import vxi11


class _Timeout:
    """Helper class for setting scoped timeout values."""

    def __init__(self, instr, timeout):
        self._instr = instr
        self._orig_timeout = self._instr.timeout
        self._instr.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._instr.timeout = self._orig_timeout


class WaveRunner:
    """Class for capturing traces using a WaveRunner oscilloscope.

    This class operates the oscilloscope in sequence mode to improve performance and in
    2-channel mode to maximize sampling rate. Trigger and power signals must be
    connected to channels 2 and 3, respectively.

    When in sequence mode, the oscilloscope captures a total of `num_segments` waves
    each starting at a trigger event. This is much more efficient than sending a
    separate command for each wave.

    This class is only tested to work with WaveRunner 9104 series.

    For more information on the commands used in this module please see:
    - Operator's Manual WaveRunner 9000 and WaveRunner 8000-R Oscilloscopes
        (http://cdn.teledynelecroy.com/files/manuals/waverunner-9000-operators-manual.pdf)
    - MAUI Oscilloscopes Remote Control and Automation Manual
        (http://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf)

    Typical usage:
    >>> waverunner = WaveRunner('192.168.1.227')
    >>> waverunner.configure()
    >>> while foo:
    >>>     ...
    >>>     waverunner.num_segments = num_segments
    >>>     waverunner.arm()
    >>>     waves = waverunner.wait_for_acquisition_and_transfer_waves()
    >>>     ...

    Attributes:
        num_segments_max: Maximum number of waves per sequence.
        num_segments: Number of waves per sequence.
        num_segments_actual: Equal to ``num_segments``.
    """

    def __init__(self, ip_addr):
        """Inits a WaveRunner.

        Connects to the oscilloscope, populates and prints device information.

        Args:
            ip_addr: IP address of the oscilloscope.
        """
        self._ip_addr = ip_addr
        self.num_segments = 1000
        self._num_samples = 740
        self._instr = vxi11.Instrument(self._ip_addr)
        self._populate_device_info()
        self._print_device_info()
        # Commented out since default configuration is highly specific
        # Class will be used more general
        # self._configure()

    @property
    def num_segments_max(self):
        return 2000

    @property
    def num_segments_actual(self):
        return self.num_segments

    def _write(self, cmd):
        self._instr.write(cmd)

    def _ask(self, cmd):
        return self._instr.ask(cmd)

    def _ask_raw(self, cmd):
        return self._instr.ask_raw(cmd)

    def _get_and_print_cmd_error(self):
        """Get command error status for last command. On error, displays error message."""
        # from p.328
        # https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf
        # Note: beware of numbering; idx starts at 0
        error_msg = ["OK.",
                     "Unrecognized command/query header.",
                     "Illegal header path.",
                     "Illegal number.",
                     "Illegal number suffix.",
                     "Unrecognized keyword.",
                     "String error.",
                     "GET embedded in another message.",
                     "",
                     "",
                     "Arbitrary data block expected.",
                     "Non-digit character in byte count field of arbitrary data block.",
                     "EOI detected during definite length data block transfer.",
                     "Extra bytes detected during definite length data block transfer"]
        return_code = int((re.findall(r'\d+', self._ask("CMR?")))[0])
        if return_code > 13 or return_code in [8, 9]:
            return_msg = f"{return_code} Unkown error code"
        else:
            return_msg = f"{return_code} {error_msg[return_code]}"
        if return_code != 0:
            print("WAVERUNNER: ERROR in last command: " + return_msg)

    def _fetch_file_from_scope(self, file_name_scope):
        fetched_file = self._ask("TRANSFER_FILE? DISK,HDD,FILE," + file_name_scope)
        # remove echoed command characters from beginning
        fetched_file = fetched_file[5:]
        self._get_and_print_cmd_error()
        return fetched_file

    def _write_to_file_on_scope(self, file_name_scope, data):
        self._write("TRANSFER_FILE DISK,HDD,FILE," + file_name_scope + "," + data)
        self._get_and_print_cmd_error()

    def _delete_file_on_scope(self, file_name_scope):
        self._write("DELETE_FILE DISK,HDD,FILE," + file_name_scope)
        self._get_and_print_cmd_error()

    def _store_setup_to_file_on_scope(self, file_name_scope):
        self._write("STORE_PANEL DISK,HDD,FILE," + file_name_scope)
        self._get_and_print_cmd_error()

    def _recall_setup_from_file_on_scope(self, file_name_scope):
        self._write("RECALL_PANEL DISK,HDD,FILE," + file_name_scope)
        self._get_and_print_cmd_error()

    def save_setup_to_local_file(self, file_name_local):
        setup_data = self._ask("PANEL_SETUP?")
        self._get_and_print_cmd_error()
        # remove echoed command characters from beginning
        setup_data = setup_data[5:]
        local_file = open(file_name_local, "w", encoding='utf-8')
        local_file.write(setup_data)
        local_file.close()

    def load_setup_from_local_file(self, file_name_local):
        # Note: Preserve line endings so that lenght matches
        # File probably received/stored from Windows scope with \r\n
        local_file = open(file_name_local, "r", newline='')
        data_read_from_file = local_file.read()
        file_name_scope = "'D:\\Temporary_setup.lss'"
        self._write_to_file_on_scope(file_name_scope, data_read_from_file)
        self._recall_setup_from_file_on_scope(file_name_scope)

    def _populate_device_info(self):
        manufacturer, model, serial, version = re.match(
            "([^,]*),([^,]*),([^,]*),([^,]*)", self._ask("*IDN?")
        ).groups()
        opts = ", ".join(self._ask("*OPT?").split(","))
        self._device_info = {
            "manufacturer": manufacturer,
            "model": model,
            "serial": serial,
            "version": version,
            "opts": opts,
        }

    def _print_device_info(self):
        def print_info(manufacturer, model, serial, version, opts):
            # TODO: logging
            print(f"Connected to {manufacturer} {model} (ip: {self._ip_addr}, serial: {serial}, "
                  f"version: {version}, options: {opts})")
            if opts == "WARNING : CURRENT REMOTE CONTROL INTERFACE IS TCPIP":
                print("ERROR: WAVERUNNER: Must set remote control to VXI11 on scope under: "
                      "Utilities > Utilities Setup > Remote")
        print_info(**self._device_info)

    def _default_setup(self):
        self._instr.timeout = 10
        # Reset the app and wait until it's done.
        self._write("vbs 'app.settodefaultsetup'")
        self._ask("vbs? 'return=app.WaitUntilIdle(1)'")
        # Prioritize analysis over rendering in the app.
        self._write("vbs 'app.Preferences.Performance = \"Analysis\"'")
        commands = [
            # Stop the trigger.
            "TRMD STOP",
            # Hide all traces for better performance.
            "C1:TRA OFF",
            "C2:TRA OFF",
            "C3:TRA OFF",
            "C4:TRA OFF",
            # Single grid.
            "GRID SINGLE",
            # Use shorter responses.
            "CHDR OFF",
            # Wait until all operations are complete.
            "*OPC?",
        ]
        res = self._ask(";".join(commands))
        assert res == "1"

    def _configure_power_channel(self):
        commands = [
            # DC coupling, 1 Mohm.
            "C3:CPL D1M",
            "C3:VDIV 35MV",
            "C3:OFST 105MV",
        ]
        self._write(";".join(commands))
        self._write("vbs 'app.Acquisition.C3.BandwidthLimit = \"200MHz\"'")
        # Noise filtering - reduces bandwidth.
        self._write("vbs 'app.Acquisition.C3.EnhanceResType = \"2.5bits\"'")

    def _configure_trigger_channel(self):
        commands = [
            # DC coupling, 1 Mohm.
            "C2:CPL D1M",
            # 0.75 V/div, -1.75 V offset.
            "C2:VDIV 0.75V",
            "C2:OFST -1.75V",
        ]
        self._write(";".join(commands))
        self._write("vbs 'app.Acquisition.C2.BandwidthLimit = \"200MHz\"'")

    def _configure_trigger(self):
        commands = [
            # Select trigger: edge, channel 2, no hold-off.
            "TRSE EDGE,SR,C2,HT,OFF",
            # Rising edge.
            "C2:TRSL POS",
            # DC coupling.
            "C2:TRCP DC",
            # Trigger level.
            "C2:TRLV 1.5",
        ]
        self._write(";".join(commands))

    def _configure_timebase(self):
        commands = [
            "TDIV 800NS",
            # Trigger delay: Trigger is centered by default. Move to the left to
            # include the samples that we are interested in.
            # Note: This number is tuned to align WaveRunner traces with ChipWhisperer
            # traces.
            "TRDL -4950NS",
        ]
        self._write(";".join(commands))

    def _configure_acquisition(self):
        # Only use channels 2 and 3 to maximize sampling rate.
        self._write("vbs 'app.Acquisition.Horizontal.ActiveChannels = \"2\"'")
        self._write("vbs 'app.Acquisition.Horizontal.Maximize = \"FixedSampleRate\"'")
        self._write("vbs 'app.Acquisition.Horizontal.SampleRate = \"1 GS/s\"'")

    def _configure_waveform_transfer(self):
        commands = [
            # SP: decimation, 10 for 1 GS/s -> 100 MS/s.
            # NP: number of points, self._num_samples.
            # FP: first point (without decimation).
            # SN: All sequences: 0
            f"WFSU SP,10,NP,{self._num_samples},FP,10,SN,0",
            # Data format: with DEF9 header, bytes (8-bit signed integers), binary encoding.
            # TODO: byte vs. word.
            "CFMT DEF9,BYTE,BIN",
            # LSB first.
            "CORD LO",
        ]
        self._write(";".join(commands))

    def _configure(self):
        """Configures the oscilloscope for acquisition."""
        self._default_setup()
        self._configure_power_channel()
        self._configure_trigger_channel()
        self._configure_trigger()
        self._configure_timebase()
        self._configure_acquisition()
        self._configure_waveform_transfer()

    def arm(self):
        """Arms the oscilloscope in sequence mode."""
        commands = [
            f"SEQ ON,{self.num_segments}",
            "TRMD SINGLE",
            "*OPC?",
        ]
        res = self._ask(";".join(commands))
        assert res == "1"

    def _parse_waveform(self, data):
        # Packet format: DAT1,#9000002002<SAMPLES>
        len_ = int(data[7:16])
        # Note: We use frombufer to minimize processing overhead.
        waves = np.frombuffer(data, np.int8, int(len_), 16)
        waves = waves.reshape((self.num_segments, self._num_samples))
        return waves

    def capture_and_transfer_waves(self):
        """Waits until the acquisition is complete and transfers waveforms.

        Returns:
            Waveforms.
        """
        # Don't process commands until the acquisition is complete and wait until
        # processing is complete.
        res = self._ask("WAIT 10;*OPC?")
        assert res == "1"
        # Transfer and parse waveform data.
        data = self._ask_raw(b"C3:WF? DAT1")
        waves = self._parse_waveform(data)
        return waves

    def display_message(self, msg):
        """Displays a message on the oscilloscope."""
        self._write(f'MSG "{msg}"')

    def buzz(self):
        """Activates the built-in buzzer."""
        self._write("BUZZ BEEP")
