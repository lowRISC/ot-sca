# Getting Started

## Prerequisites

### Hardware Requirements

To perform side-channel analysis (SCA) for [OpenTitan](https://github.com/lowRISC/OpenTitan)
using the infrastructure provided in this repository, the following hardware
equipment is required:

* [ChipWhisperer CW310 FPGA Board with K410T](https://rtfm.newae.com/Targets/CW310%20Bergen%20Board//)

  This is the target board. The Xilinx Kintex-7 FPGA is used to implement
  the full Earl Grey top level of OpenTitan. Note that there are different
  versions of the board available of which the board with the bigger FPGA
  device (K410T) is required.

* [ChipWhisperer-Husky CW1190 Capture Board](https://github.com/newaetech/ChipWhisperer-Husky)

  This is the capture or scope board. It is used to capture power traces of
  OpenTitan implemented on the target board. The capture board comes with
  SMA and 20-pin ChipWhisperer cable required to connect to the target board.

The following, alternative hardware equipment is no longer supported:

* [ChipWhisperer CW305-A100 FPGA Board](https://rtfm.newae.com/Targets/CW305%20Artix%20FPGA/)

  This is a smaller target board than the CW310.

  **Note:** The Xilinx Artix-7 A100T FPGA on the CW305-A100 is considerably
  smaller than the Kintex-7 K410T FPGA on the CW310 and does not offer
  sufficient resources for implementing the full Earl Grey top level of
  OpenTitan. However, it is suitable for implementing the English Breakfast
  top level, a substantially reduced version of OpenTitan. As a result, the
  CW305-A100 is suitable for performing SCA of AES but not KMAC or OTBN. Note
  that also for this board there exist different versions of which of which
  only the board with the bigger FPGA device (XC7A100T-2FTG256) is supported.

  **Support:** Older versions of the repo support this target.

### Hardware Setup

To setup the hardware, first connect the target and capture board together. You
don't want to accidentally short something out with the conductive SMA
connector.

#### CW310

As shown in the photo below,
1. Connect the `SHUNTLOW AMPLIFIED` output of the CW310 (topmost SMA) to the
   CW-Husky `Pos` input.
1. Connect the `ChipWhisperer Connector` (`J14`) on the CW310 to the
   `ChipWhisperer 20-Pin Connector` on the capture board. On the CW-Husky this
   is the connector on the *SIDE* not the connector on the *FRONT*.

![](img/cw310_cwhusky.jpg)

The CW310 will need a power supply, the default power supply uses a DC barrel
connector to supply 12V. If you use this power supply all switches should be
in the default position. To this end, make sure to:
1. Set the `SW2` switch (to the right of the barrel connector) up to the
   `5V Regulator` option.
1. Set the switch below the barrel connector to the right towards the `Barrel`
   option.
1. Set the `Control Power` switch (bottom left corner, `SW7`) to the right.
1. Ensure the `Tgt Power` switch (above the fan, `S1`) is set to the right
   towards the `Auto` option.

Then,
1. Plug the DC power adapter into the barrel jack (`J11`) in the top left
   corner of the board.
1. Use a USB-C cable to connect your PC with the `USB-C Data` connector (`J8`)
   in the lower left corner on the board.
1. Connect the CW-Husky to your PC via USB-C cable.

You should see the blue "Status" LED on the CW310 blinking, along with several
green power LEDs. The "USB-C Power" led may be red as there is no USB-C PD
source. The CW-Husky should also have a green blinking status LED at this
point. If LEDs are solid it may mean the device has not enumerated, which might
require additional setup (see UDEV Rules below).

### UDEV Rules

You might need to setup the following `udev` rules to gain access to the two
USB devices. To do so, open the file `/etc/udev/rules.d/90-lowrisc.rules` or
create it if does not yet exist and add the following content to it:

```
# CW310 (Kintex Target)
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="c310", MODE="0666" SYMLINK+="opentitan/cw_310"

# CW-Husky
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="ace5", MODE="0666" SYMLINK+="opentitan/cw_husky"
```

To activate the rules, type
```console
$ sudo udevadm control --reload
```
and then disconnect and reconnect the devices.

With the optional `SYMLINK` attribute, a stable symbolic link is created,
helping to identify the USB devices more easily.

For more details on how to set up `udev` rules, see the corresponding section
in the [OpenTitan documentation](https://docs.opentitan.org/doc/ug/install_instructions/#xilinx-vivado).

### Software Requirements

Software required by this repository can either be directly installed on a
machine or obtained using the provided [Dockerfile](https://github.com/lowRISC/ot-sca/blob/master/util/docker/Dockerfile).


#### Installing on a Machine

##### Python Virtual Environment

To avoid potential dependency conflicts (in particular with the main OpenTitan
repository which may be using a different version of ChipWhisperer) it is
strongly recommended to setup a virtual Python environment for this repository.

To this end, type
```console
$ python3 -m venv .venv
```
to create a new virtual environment.

From this point on, whenever you want to work with this repository, type
```console
$ source .venv/bin/activate
```
to enable the virtual environment previously initialized.


##### Python Dependencies

This repository has a couple of Python dependencies to be installed through
`pip`. It is recommended to first install the latest version of `pip` and
`setuptools` using
```console
python3 -m pip install -U pip "setuptools<66.0.0"
```
You can then run
```console
$ pip install -r python-requirements.txt
```
to install all the required Python dependencies.


##### ChipWhisperer Dependencies

ChipWhisperer itself is installed as a Python dependency through
`python_requirements.txt`.

However, it has some non-Python dependencies related to USB. Please see
[this page](https://chipwhisperer.readthedocs.io/en/latest/linux-install.html#required-packages)
to install the specific `apt` packages required by ChipWhisperer.

**Notes:**
- We recommend to use Python 3.10. Later versions might also work.
- CW-Husky requires ChipWhisperer 5.7.0 or later. The default
  `python_requirements.txt` will install a version supporting CW-Husky.

##### Git Large File Storage (LFS)

This project uses Git LFS for storing binaries like example target FPGA
bitstreams on a remote server. The repository itself just contains diff-able
text pointers to the binaries. It is recommended to install the `git-lfs`
tool for transparently accessing the binaries. Alternatively, they can be
downloaded manually from GitHub.

You can run
```console
$ sudo apt install git-lfs
```
to install the `git-lfs` tool on your Ubuntu machine.

If you've cloned the `ot-sca` repo before installing Git LFS, run
```console
$ git-lfs pull
```
to get the binaries.

Alternatively, you can rebuild those binaries yourself from the
[OpenTitan](https://github.com/lowRISC/OpenTitan) repository.


#### Using the Docker Image

**Notes**:
- While the provided container image can be run using the Docker Engine, this
  getting started guide relies on Podman instead to avoid requiring root
  permissions on the host.
- This is a WIP and currently supports Linux hosts only.

The
[Dockerfile](https://github.com/lowRISC/ot-sca/blob/master/util/docker/Dockerfile)
in this repository can be used to build a ready-to-use image with all the
dependencies installed. To build the image:
1. If not already installed, install Podman and containers-storage following the instructions
[here](https://podman.io/getting-started/installation), and
2. Build the container image using
[build\_image.sh](https://github.com/lowRISC/ot-sca/blob/master/util/docker/build_image.sh):
```console
$ util/docker/build_image.sh
```

Once the image is built, you can run it using
[run\_container.sh](https://github.com/lowRISC/ot-sca/blob/master/util/docker/run_container.sh):
```console
$ util/docker/run_container.sh -h

Run OpenTitan SCA/FI image.

Usage: util/docker/run_container.sh -d DEVICE [-d DEVICE] -m SHM_SIZE -w HOST_WORK_DIR [-h]

  -d: Host device to be added to the container. This option can be used multiple times.
  -m: Shared memory size (/dev/shm) of the container. Should be at least 1/3 of total memory.
  -w: Host directory that will be mounted into the container as /repo
  -h: Print usage information and exit.
```

For example, if the host has 32+ GB RAM, the ot-sca repository is at
`~/repos/ot-sca`, and ChipWhisperer devices are at `/dev/opentitan/cw_310`
and `/dev/opentitan/cw_husky`, you can use:
```console
$ util/docker/run_container.sh -d /dev/opentitan/cw_310 -d /dev/opentitan/cw_husky -m 12g -w ~/repos/ot-sca
```

If ChipWhisperer devices are the only USB devices that are connected to the
host, you can use the following to add all USB devices to the container:
```console
$ util/docker/run_container.sh $(find /dev/opentitan -type l -exec readlink -f {} \; | sed 's/^/-d /g' | xargs echo) -m 12g -w ~/repos/ot-sca
```

Once the container is running, try capturing some traces to verify that
everything is working correctly:
```console
Creating user 'ot' with UID=1000, GID=1000.
ot@ot-sca:/repo$ cd cw/
ot@ot-sca:/repo/cw$ ./capture.py --cfg-file capture_aes_cw310.yaml capture aes-random
Connecting and loading FPGA... Done!
Initializing PLL1
Programming OpenTitan with "objs/aes_serial_fpga_cw310.bin"...
Transferring frame 0x00000000 @             0x00000000.
Transferring frame 0x00000001 @             0x000007D8.
Transferring frame 0x00000002 @             0x00000FB0.
Transferring frame 0x00000003 @             0x00001788.
Transferring frame 0x00000004 @             0x00001F60.
Transferring frame 0x00000005 @             0x00002738.
Transferring frame 0x80000006 @             0x00002F10.
Scope setup with sampling rate 200009376.0 S/s
Reading from FPGA using simpleserial protocol.
Target simpleserial version: z01 (attempts: 1).
Using key: b'2b7e151628aed2a6abf7158809cf4f3c'
Capturing: 100%|████████████████████████████| 5000/5000 [00:55<00:00, 90.34it/s]
Created plot with 10 traces: ~/ot-sca/cw/projects/sample_traces_aes.html
```


### Generating OpenTitan Binaries

Instead of using the example binaries provided by this repository via Git LFS,
you can regenerate them from the
[OpenTitan](https://github.com/lowRISC/OpenTitan) repository.

Below, we quickly outline the necessary steps for building the binaries for the
CW310 board from the sources. For more information on the build
steps, refer to the [OpenTitan FPGA documentation](https://docs.opentitan.org/doc/ug/getting_started_fpga/).

Note that below we use `$REPO_TOP` to refer to the main OpenTitan repository
top, not this `ot-sca` repo.

#### Earl Grey for CW310

To build FPGA bitstreams and penetration testing framework binaries for CW310, please follow this [guide](./building_fpga_bitstreams.md).

## Configuration

The main configuration of the OpenTitan SCA setup is stored in the files
```
capture/configs/aes_sca_cw310.yaml
```
for AES.

For example, these files allow to specify the FPGA bitstream to be loaded, the
OpenTitan application binary to execute, the default number of traces to
capture and the ADC gain. By default, the prebuilt binaries delivered with this
repository are used. If you want to use custom binaries, open the file and edit
the specified file paths.

**Notes**:
* When working with custom binaries, it is recommended to always re-generate
  both bitstream and application binaries from the OpenTitan repository.
  Otherwise you might risk to end up with an incompatible combination of
  bitstream and application binary.
* The default configurations target the CW310 board.


## Capturing Power Traces

Our setup supports capturing AES, KMAC, SHA3 and OTBN power traces using a CW-Husky capture board.
The setup also supports capturing power traces in batches to achieve substantially higher capture rates.

Before starting a long running capture, it is recommended to always perform a capture with fewer traces to make sure the setup is configured as expected (e.g. ADC gain).

### AES Capture

To perform a capture `capture/capture_aes.py` is used with default configurations set in the `*.yaml` files.
When using this script configuration file and project path need to be given as parameters.

There are 4 capture modes for AES:
1. AES Random
1. AES Random Batch
1. AES Fixed vs Random Key
1. AES Fixed vs Random Key Batch

Modes `aes_fvsr_key_batch` and `aes_fvsr_key` capture data for fixed-vs-random key analysis.
This analysis is described in DTR TVLA Section 5.3. "General Test: Fixed-vs.-Random Key Datasets".
In this data set two types of traces are collected, namely fixed (using a constant key) and random (using a randomly selected key).
Each measurement uses a randomly generated plaintext.
Measurements alternate between fixed and random, in a random manner (depending on a randomly generated bit).

Modes `aes_random_batch` and `aes_random` capture data for random data analysis.
In this analysis, the key is constant for all measurements, and all plaintexts are randomly generated.

The capture type can be configured by setting the `which test:` parameter in the configuration file.
This can be done by uncommenting the corresponding line.
```
  # which_test: aes_random_batch
  # which_test: aes_random
  which_test: aes_fvsr_key_batch
  # which_test: aes_fvsr_key
```

To perform AES capture using any mode, you can use the following command:
```console
$ cd capture
$ ./capture_aes.py -c configs/aes_sca_cw310.yaml -p projects/aes_sca_cw310
```

This script will load the OpenTitan FPGA bitstream to the target board, load and start the application binary to the target via SPI, and then feed data in and out of the target while capturing power traces on the capture board.
It should produce console output similar to the following output:

```console
Initializing target cw310 ...
Connecting and loading FPGA... Done!
Initializing PLL1
Programming 256 bytes at address 0x00000000.
Programming 256 bytes at address 0x00000100.
Programming 256 bytes at address 0x00000200.
...
Programming 112 bytes at address 0x00009000.
Target simpleserial version: z01 (attempts: 1).
INFO:root:Initializing scope husky with a sampling rate of 200000000...
Initializing scope husky with a sampling rate of 200000000...
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:378) ADC frequency must be between 1MHz and 300000000.0MHz - ADC mul has been adjusted to 2
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:409) PLL unlocked after updating frequencies
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:410) Target clock has dropped for a moment. You may need to reset your target
Connected to ChipWhisperer (num_samples: 501, num_samples_actual: 501, num_segments_actual: 1)
INFO:root:Setting up capture aes_random batch=True...
Setting up capture aes_random batch=True...
INFO:root:Initializing OT AES with key b'811e3731b0120a7842781e22b25cddf9' ...
Initializing OT AES with key b'811e3731b0120a7842781e22b25cddf9' ...
Capturing: 100%|██████████████████████| 1000/1000 [00:01<00:00, 807.02 traces/s]
```

In case you see console output like
```console
WARNING:root:Your firmware is outdated - latest is 0.20. Suggested to update firmware, as you may experience errors
See https://chipwhisperer.readthedocs.io/en/latest/api.html#firmware-update
```
you should update the firmware of one or both of the ChipWhisperer boards.
This process is straightforward and well documented online.
Simply follow the link printed on the console.

Once the power traces have been collected, a picture similar to the following should be shown in your browser.

![](img/aes_traces.jpg)

Note the input data can range from `0` to `4095` (this is because ADC resolution on CW-Husky is 12 bits).
If the output value is exceeding those limits you are clipping and losing data.
But if the range is too small you are not using the full dynamic range of the ADC.
You can tune the `scope_gain` setting in the `.yaml` configuration file.
Note that boards have some natural variation, and changes such as the clock frequency, core voltage, and device utilization (FPGA build) will all affect the safe maximum gain setting.

### KMAC-128 Capture

There are 3 capture modes for KMAC:
1. KMAC Random
1. KMAC Fixed vs Random Key
1. KMAC Fixed vs Random Key Batch

The capture type can be configured by setting the `which test:` parameter in the configuration file.
This can be done by uncommenting the corresponding line.
```
  # which_test: kmac_random
  # which_test: kmac_fvsr_key
  which_test: kmac_fvsr_key_batch
```

To perform a KMAC-128 capture, use this command:
```console
$ cd capture
$ ./capture_kmac.py -c configs/kmac_sca_cw310.yaml -p projects/kmac_sca_cw310
```


You should see similar output as in the AES example.
Once the power traces have been collected, a picture similar to the following should be shown in your browser.

![](img/kmac_traces.jpg)

### SHA3 Capture

There are 3 capture modes for SHA3:
1. SHA3 Random
1. SHA3 Fixed vs Random Data
1. SHA3 Fixed vs Random Data Batch

To perform a SHA3 capture, use this command:
```console
$ cd capture
$ ./capture_sha3.py -c configs/sha3_sca_cw310.yaml -p projects/sha3_sca_cw310
```

You should see similar output as in the KMAC example. Once the power traces have
been collected, a picture similar to the following should be shown in your
browser.

![](img/sha3_traces.jpg)


You can also use the batch mode with
```console
$ cd cw
$ ./capture.py --cfg-file capture_sha3_cw310.yaml capture sha3-fvsr-data-batch --num-traces 100 --plot-traces 10
```

Errors might occur due to synchronization issues. In this case try to reset the board and
try again.

To disable masking use the ```sha3_sca_cw310_masks_off_cw310.yaml``` file. The capture commands stay the same.



### SHA3 TVLA

Currently, we only support the general (non-specific) TVLA, which should only be used on a Fixed-vs.-Random
Dataset. To perform the analysis for the masked version, run:

```console
$ cd cw
$ ./tvla.py --cfg-file tvla_cfg_sha3.yaml run-tvla
```

For disabled masking run:
```console
$ cd cw
$ ./tvla.py --cfg-file tvla_cfg_sha3_masks_off.yaml run-tvla
```


## Performing Example SCA Attack on AES with Masking Disabled

The OpenTitan AES module uses boolean masking to aggravate SCA attacks. For
evaluating the SCA setup as well as specific attacks, the OpenTitan AES
module supports an FPGA-only compile-time Verilog parameter allowing software
to disable the masking. In the following example, we are going to disable the
masking in order to demonstrate an example attack. Note that On the ASIC, the
masking cannot be disabled.

### Disabling Masking

1. In the OpenTitan repository, edit the file `sw/device/sca/aes_serial.c` to
   and change `.masking = kDifAesMaskingInternalPrng` to
   `.masking = kDifAesMaskingForceZero`.
1. From the root directory of the OpenTitan repository, run
```console
$ ./bazelisk.sh build //sw/...
```
   To re-generate the application binary.

### Configuration

Open the config file `cw/capture_aes_cw310.yaml` and make sure to:
1. Change the path to application binary to use the one from the previous step.
1. If necessary, change the path to the bitstream to use a bitstream matching
   The application binary.

### Capturing Power Traces

Run the following commands

```console
$ cd cw
$ ./capture.py --cfg-file capture_aes_cw310.yaml capture aes-random-batch --num-traces 2000000
```
to start the capture.

### Performing the Attack

Run the following commands:

```console
$ cd cw
$ ./ceca.py -f projects/opentitan_simple_aes.cwp -a 505 520 -d output -s 3 -w 16 -n 2000000
```

This should give you an output like this:

```console
2021-10-23 13:19:19,846 INFO ceca.py:520 -- Will use 1962961 traces (98.1% of all traces)
2021-10-23 13:19:21,115 INFO _timer.py:61 -- compute_pairwise_diffs_and_scores took 0.4s
2021-10-23 13:19:21,116 INFO ceca.py:528 -- Difference values (delta_0_i): [  0 196  41 120  25  62 245  89  49 239 220  24 102 179 220 118]
2021-10-23 13:19:21,158 INFO ceca.py:532 -- Recovered AES key: 2b7e151628aed2a6abf7158809cf4f3c
2021-10-23 13:19:21,159 INFO ceca.py:538 -- Recovered 82/120 differences between key bytes
2021-10-23 13:19:21,159 INFO _timer.py:61 -- perform_attack took 13.6s
2021-10-23 13:19:21,159 INFO _timer.py:61 -- main took 14.7s
```

The setting of `-n 2000000` is how many traces to use, if you receive a memory error reduce
the number of traces.

The setting of `-a 505 520` specifies a location in the power traces, you may need to change
these settings with new FPGA builds as the leakage location will shift.

### Debugging

Run the following command to see serial outputs of the target program:

```console
$ screen /dev/ttyACM1 115200,cs8,-ixon,-ixoff
```
Note that you may need to change "ttyACM1" to something else.

# Trace Preprocessing

The `analysis/trace_preprocessing.py` script allows the user to preprocess traces that were
captured with the OpenTitan trace library. This script provides two functionalities:
- Trace filtering: Remove traces that contain samples over the tolerable deviation from average.
- Trace aligning: Uses a trace window to align all traces according to a reference trace.

The filtering can be enabled with the `-f` command line argument and the tolerable deviation
from average with `-s`.
To turn on the trace aligning mechanism, use the `-a` flag. The reference trace is calculated
from the mean of `-n` traces and the window can be specified with `-lw` and `-hw`.
When operating with larger databases, the `-c` parameter can be used to specifiy the number
of processes used for aligning the traces and the `-m` parameter can be used to specify the
maximum amount of traces that are kept in memory per process.
An example is shown below:

```console
$ ./trace_preprocessing.py -i db_in.db -o db_out.db -f -s 7.5 -a -p 5 -n 1000 -lw 21000 -hw 23000 -ms 20 -m 1000 -c 4
```

# Troubleshooting

## Unreachable Husky Scope
If a connection with CW-Husky cannot be established, e.g., due to a failed
firmware update, first try to [erase](https://rtfm.newae.com/Capture/ChipWhisperer-Husky/#erase-pins)
its firmware by shorting ```SJ1``` on the board. If the scope is still
unreachable but the USB connection is detected by Linux, load the firmware
manually:
```python
import chipwhisperer as cw
programmer = cw.SAMFWLoader(scope=None)
programmer.program('/dev/ttyACM0', hardware_type='cwhusky')
```
Replace ```/dev/ttyACM0``` with the corresponding device.
