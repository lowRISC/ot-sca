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

    This class can be used to operate the oscilloscope in sequence mode. The
    configuration can be done manually and loaded from a file using provided
    functions. In sequence mode, the oscilloscope captures a total of
    `num_segments` waves each starting at a trigger event. This is much more
    efficient than sending a separate command for each wave.

    This class is only tested with WaveRunner 9104 series.

    For more information on the commands used in this module please see:
    - Operator's Manual WaveRunner 9000 and WaveRunner 8000-R Oscilloscopes
        (http://cdn.teledynelecroy.com/files/manuals/waverunner-9000-operators-manual.pdf)
    - MAUI Oscilloscopes Remote Control and Automation Manual
        (http://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf)

    Typical usage with default configuration:
    >>> waverunner = WaveRunner('192.168.1.227')
    >>> waverunner.load_setup_from_local_file(file_name_local)
    >>> waverunner.configure_waveform_transfer_general(num_segments=10, sparsing=1, \\
            num_samples=1000, first_point=0)
    >>> while foo:
    >>>     ...
    >>>     waverunner.arm()
    >>>     waves = waverunner.capture_and_transfer_waves()
    >>>     ...

    The class also provides a default configuration through functions. Then,
    the trigger and power signals must be connected to channels 2 and 3. Note
    that the configuration is through hard-coded parameters in this file!

    Typical usage with default configuration:
    >>> waverunner = WaveRunner('192.168.1.227')
    >>> waverunner.configure()
    >>> while foo:
    >>>     ...
    >>>     waverunner.num_segments = num_segments
    >>>     waverunner.arm()
    >>>     waves = waverunner.capture_and_transfer_waves()
    >>>     ...

    Attributes:
        num_segments: Number of segments per sequence.
        num_samples: Number of samples per segment.
    """

    def __init__(self, ip_addr):
        """Inits a WaveRunner.

        Connects to the oscilloscope, populates and prints device information.

        Args:
            ip_addr: IP address of the oscilloscope.
        """
        self._ip_addr = ip_addr
        self.num_segments = 1000
        self.num_samples = 740
        self._instr = vxi11.Instrument(self._ip_addr)
        self._populate_device_info()
        self._print_device_info()
        self.acqu_channel = "C3"

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
        # Note: Scope error log under Utilities/Remote/Show Remote Control Log
        # Note: Beware of numbering; idx starts at 0
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
        print("WAVERUNNER: Saving setup to " + file_name_local)
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
        print("WAVERUNNER: Loading setup from " + file_name_local)
        local_file = open(file_name_local, "r", newline='', encoding='utf-8')
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
            print(f"WAVERUNNER: Connected to {manufacturer} {model} (ip: "
                  f"{self._ip_addr}, serial: {serial}, "
                  f"version: {version}, options: {opts})")
            if opts == "WARNING : CURRENT REMOTE CONTROL INTERFACE IS TCPIP":
                print("ERROR: WAVERUNNER: Must set remote control to VXI11 on scope under: "
                      "Utilities > Utilities Setup > Remote")
        print_info(**self._device_info)

    def _default_setup(self):
        # Note this is a default configuration and might not be meaningfull always
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
            f"{self.acqu_channel}:TRA OFF",
            "C4:TRA OFF",
            # Single grid.
            "GRID SINGLE",
            # Use shorter responses.
            "CHDR OFF",
            # Wait until all operations are complete.
            "*OPC?",
        ]
        res = self._ask(";".join(commands))
        assert res == "*OPC 1"

    def _configure_power_channel(self):
        # Note this is a default configuration and might not be meaningfull always
        commands = [
            # DC coupling, 1 Mohm.
            f"{self.acqu_channel}:CPL D1M",
            f"{self.acqu_channel}:VDIV 35MV",
            f"{self.acqu_channel}:OFST 105MV",
        ]
        self._write(";".join(commands))
        self._write(f"vbs 'app.Acquisition.{self.acqu_channel}.BandwidthLimit = \"200MHz\"'")
        # Noise filtering - reduces bandwidth.
        self._write(f"vbs 'app.Acquisition.{self.acqu_channel}.EnhanceResType = \"2.5bits\"'")

    def _configure_trigger_channel(self):
        # Note this is a default configuration and might not be meaningfull always
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
        # Note this is a default configuration and might not be meaningfull always
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
        # Note this is a default configuration and might not be meaningfull always
        commands = [
            "TDIV 800NS",
            # Trigger delay: Trigger is centered by default. Move to the left to
            # include the samples that we are interested in.
            # Note: This number is tuned to align WaveRunner traces with ChipWhisperer
            # traces.
            "TRDL -4950NS",
        ]
        self._write(";".join(commands))

    def _configure_acqu(self):
        # Note this is a default configuration and might not be meaningfull always
        # Only use channels 2 and 3 to maximize sampling rate.
        self._write("vbs 'app.Acquisition.Horizontal.ActiveChannels = \"2\"'")
        self._write("vbs 'app.Acquisition.Horizontal.Maximize = \"FixedSampleRate\"'")
        self._write("vbs 'app.Acquisition.Horizontal.SampleRate = \"1 GS/s\"'")

    def _configure_waveform_transfer(self):
        # Note this is a default configuration and might not be meaningfull always
        commands = [
            # SP: decimation, 10 for 1 GS/s -> 100 MS/s.
            # NP: number of points, self.num_samples.
            # FP: first point (without decimation).
            # SN: All sequences: 0
            f"WFSU SP,10,NP,{self.num_samples},FP,10,SN,0",
            # Data format: with DEF9 header, bytes (8-bit signed integers), binary encoding.
            "CFMT DEF9,BYTE,BIN",
            # LSB first.
            "CORD LO",
        ]
        self._write(";".join(commands))

    def configure_waveform_transfer_general(self,
                                            num_segments,
                                            sparsing,
                                            num_samples,
                                            first_point,
                                            acqu_channel):
        """Configures the oscilloscope for acqu with given parameters."""
        print(f"WAVERUNNER: Configuring with num_segments={num_segments}, "
              f"sparsing={sparsing}, num_samples={num_samples}, "
              f"first_point={first_point}, acqu_channel=" + acqu_channel)
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.acqu_channel = acqu_channel
        commands = [
            # WAVEFORM_SETUP
            # SP: sparsing, e.g. 10 for every 10th point, 1 for every point.
            # NP: number of points, self.num_samples.
            # FP: first point (without sparsing).
            # SN: All sequences shall be sent: 0.
            f"WFSU SP,{sparsing},NP,{num_samples},FP,{first_point},SN,0",
            # COMM_FORMAT
            # Data format: with DEF9 header, bytes (8-bit signed integers), binary encoding.
            # TODO: We currently transfer bytes. Use WORD for larger ADCs
            "CFMT DEF9,BYTE,BIN",
            # COMM_ORDER
            # LO means LSB first.
            "CORD LO",
        ]
        self._write(";".join(commands))
        self._get_and_print_cmd_error()

    def configure(self):
        """Configures the oscilloscope for acqu with default configuration."""
        # Note this is a default configuration and might not be meaningfull always
        self._default_setup()
        self._configure_power_channel()
        self._configure_trigger_channel()
        self._configure_trigger()
        self._configure_timebase()
        self._configure_acqu()
        self._configure_waveform_transfer()

    def arm(self):
        """Arms the oscilloscope in sequence mode for selected channel."""
        # SEQ SEQUENCE Mode
        # TRMD Trigger Mode Single
        commands = [
            f"SEQ ON,{self.num_segments}",
            "TRMD SINGLE",
            "*OPC?",
        ]
        res = self._ask(";".join(commands))
        assert res == "*OPC 1"

    def _parse_waveform(self, data):
        # Packet format example:b'C1:WF DAT1,#900002002<SAMPLES>
        len_ = int(data[13:22])
        # Note: We use frombufer to minimize processing overhead.
        waves = np.frombuffer(data, np.int8, int(len_), 22)
        waves = waves.reshape((self.num_segments, self.num_samples))
        return waves

    def capture_and_transfer_waves(self):
        """Waits until the acqu is complete and transfers waveforms.

        Returns:
            Waveforms.
        """
        # Don't process commands until the acqu is complete and wait until
        # processing is complete.
        res = self._ask("WAIT 10;*OPC?")
        assert res == "*OPC 1"
        # Transfer and parse waveform data.
        if self.acqu_channel == "C1":
            data = self._ask_raw(b"C1:WF? DAT1")
        elif self.acqu_channel == "C2":
            data = self._ask_raw(b"C2:WF? DAT1")
        elif self.acqu_channel == "C3":
            data = self._ask_raw(b"C3:WF? DAT1")
        elif self.acqu_channel == "C4":
            data = self._ask_raw(b"C4:WF? DAT1")
        else:
            print("WAVERUNNER: Error: Channel selection invalid")
        waves = self._parse_waveform(data)
        return waves

    def display_message(self, msg):
        """Displays a message on the oscilloscope."""
        self._write(f'MSG "{msg}"')

    def buzz(self):
        """Activates the built-in buzzer."""
        self._write("BUZZ BEEP")
