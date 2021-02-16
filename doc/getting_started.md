# Getting Started

## Prerequisites

### Hardware Requirements

To perform side-channel analysis (SCA) for [OpenTitan](https://github.com/lowRISC/OpenTitan).
using the infrastructure provided in this repository, the following hardware
equipment is required:
* [ChipWhisperer CW305-A100 FPGA Board](https://rtfm.newae.com/Targets/CW305%20Artix%20FPGA/)
  This is the target board. The Xilinx Artix-7 FPGA is used to implement
  OpenTitan. Note that there are different versions of the board
  available of which the board with the bigger FPGA device (XC7A100T-2FTG256)
  is required.
* [ChipWhisperer-Lite CW1173 Capture Board](https://rtfm.newae.com/Capture/ChipWhisperer-Lite/)
  This is the capture or scope board. It is used to capture power traces of
  OpenTitan implemented on the target board. There are several different
  versions of the board available of which the single board without target
  processor but with SMA and 20-pin cables is required.


### Software Requirements

Software required by this repository can either be directly installed on a
machine or obtained using the provided [Dockerfile](https://github.com/lowRISC/ot-sca/blob/master/util/docker/Dockerfile).

#### Installing on a Machine

##### Python Dependencies

This repository has a couple of Python dependencies. You can run
```console
$ pip install --user -r python_requirements.txt
```
to install those dependencies.

##### ChipWhisperer Dependencies

Please see [this page](https://chipwhisperer.readthedocs.io/en/latest/prerequisites.html#packages)
to install the packages required by ChipWhisperer.

##### Git Large File Storage (LFS)

This project uses Git LFS for storing binaries like a debian-compatible
pre-built binary of the tool for loading the OpenTitan application binary over
SPI and an example target FPGA bitstream on a remote server. The repository
itself just contains diff-able text pointers to the binaries. It is recommended
to install the `git-lfs` tool for transparently accessing the binaries.
Alternatively, they can be downloaded manually from GitHub.

You can run
```console
$ sudo apt install git-lfs
```
to install the `git-lfs` tool on your Ubuntu machine.

Alternatively, you can rebuild those binaries yourself from the
[OpenTitan](https://github.com/lowRISC/OpenTitan) repository.


#### Using the Docker Image

**Note**: This is a WIP and currently supports only Linux hosts.

The
[Dockerfile](https://github.com/lowRISC/ot-sca/blob/master/util/docker/Dockerfile)
in this repository can be used to build a ready-to-use image with all the
dependencies installed. To build the image:
1. If not already installed, install the Docker Engine following the instructions
[here](https://docs.docker.com/engine/install/), and
2. Build the container image using
[build\_image.sh](https://github.com/lowRISC/ot-sca/blob/master/util/docker/run_container.sh)
(you may have to use `sudo` to be able to run docker commands depending on your setup):
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
`~/repos/ot-sca`, and ChipWhisperer devices are at `/dev/bus/usb/001/036`
and `/dev/bus/usb/001/038`, you can use:
```console
$ util/docker/run_container.sh -d /dev/bus/usb/001/036 -d /dev/bus/usb/001/038 -m 12g -w ~/repos/ot-sca
```

If ChipWhisperer devices are the only USB devices that are connected to the
host, you can use the following to add all USB devices to the container:
```console
$ util/docker/run_container.sh $(find /dev/bus/usb -type c | sed 's/^/-d /g' | xargs echo) -m 12g -w ~/repos/ot-sca
```

If you want to add only ChipWhisperer devices to the container and don't want
to search for the correct device nodes every time they are disconnected and
connected, you can add the following rules in
`/etc/udev/rules.d/90-opentitan.rules` to create stable symbolic links using
the `SYMLINK` attribute:
```
# CW-Lite
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="ace2", MODE="0666", SYMLINK+="opentitan/cw_lite"

# CW-305 (Artix Target)
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="c305", MODE="0666", SYMLINK+="opentitan/cw_305"
```

Load the new rules:
```console
$ sudo udevadm control --reload
```

Reconnect the devices and use the following to run the image in a container:
```console
$ util/docker/run_container.sh $(find /dev/opentitan -type l -exec readlink -f {} \; | sed 's/^/-d /g' | xargs echo) -m 12g -w ~/repos/ot-sca
```

Once the container is running, try capturing some traces to verify that
everything is working correctly:
```console
Creating user 'ot' with UID=1000, GID=1000.
ot@ot-sca:/repo$ cd cw/cw305/
ot@ot-sca:/repo/cw/cw305$ ./simple_capture_traces.py 
Connecting and loading FPGA... Done!
Initializing PLL1
Programming OpenTitan with "objs/aes_serial_fpga_nexysvideo.bin"...
Transferring frame 0x00000000 @ 0x00000000.
Transferring frame 0x00000001 @ 0x000003D8.
Transferring frame 0x00000002 @ 0x000007B0.
Transferring frame 0x00000003 @ 0x00000B88.
Transferring frame 0x00000004 @ 0x00000F60.
Transferring frame 0x00000005 @ 0x00001338.
Transferring frame 0x00000006 @ 0x00001710.
Transferring frame 0x00000007 @ 0x00001AE8.
Transferring frame 0x00000008 @ 0x00001EC0.
Transferring frame 0x00000009 @ 0x00002298.
Transferring frame 0x0000000A @ 0x00002670.
Transferring frame 0x8000000B @ 0x00002A48.
Serial baud rate = 38400
Serial baud rate = 115200
Scope setup with sampling rate 100003051.0 S/s
Target simpleserial version: z01 (attempts: 2).
Using key: b'2b7e151628aed2a6abf7158809cf4f3c'
Reading from FPGA using simpleserial protocol.
Checking version: 
Capturing: 100%|████████████████████████████| 5000/5000 [00:55<00:00, 90.34it/s]
```


### Generating OpenTitan Binaries

Instead of using the example binaries provided by this repository via Git LFS,
you can regenerate them from the
[OpenTitan](https://github.com/lowRISC/OpenTitan) repository.

To this end, follow these steps:

1. Go to the root directory of the OpenTitan repository.
1. Before generating the OpenTitan FPGA bitstream for the CW305 target board,
   you first have to run
```console
$ ./hw/top_englishbreakfast/util/prepare_sw.py
```
   in order to prepare the OpenTitan software build flow for the CW305 target
   board. More precisely, this script runs some code generators, patches some
   auto-generated source files and finally generates the boot ROM needed for
   bitstream generation.

   After this command has been run, the boot ROM can manually be regenerated
   by executing
```console
$ ./meson_init.sh
$ ninja -C build-out sw/device/boot_rom/boot_rom_export_fpga_nexysvideo
```
   Finally, the bitstream generation can be started by running
```console
$ fusesoc --cores-root . run --flag=fileset_topgen --target=synth lowrisc:systems:top_englishbreakfast_cw305
```
   For more information on the build steps, refer to the
   [OpenTitan FPGA documentation](https://docs.opentitan.org/doc/ug/getting_started_fpga/).

   The generated bitstream can be found in
```
build/lowrisc_systems_top_englishbreakfast_cw305_0.1/synth-vivado/lowrisc_systems_top_englishbreakfast_cw305_0.1.bit
```
   and will be loaded to the FPGA using the ChipWhisperer Python API.

1. To generate the OpenTitan application binary, make sure the
   `prepare_sw.py` script has been run before executing
```console
$ ./meson_init.sh
$ ninja -C build-out sw/device/sca/aes_serial_export_fpga_nexysvideo
```
   The generated binary can be found in
```
build-bin/sw/device/sca/aes_serial_fpga_nexysvideo.bin
```


## Setup

### Setting up Hardware

To setup the hardware, connect the two boards to your PC via USB. Make sure the
S1 jumper on the back of the target board is set to `111` such that the FPGA
bitstream can be reconfigured via USB. The target and the capture board have
further to be connected using the ChipWhisperer 20-pin connector and an SMA
cable (X4 output on target board to MEASURE input on capture board).

In addition you might need to setup the following `udev` rules to gain access
to the two USB devices. To do so, open the file
`/etc/udev/rules.d/90-lowrisc.rules` or create it if does not yet exist and add
the following content to it:

```
# NewAE Technology Inc. ChipWhisperer CW305
ACTION=="add|change", SUBSYSTEM=="usb|tty", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="c305", MODE="0666"

# NewAE Technology Inc. ChipWhisperer-Lite CW1173
ACTION=="add|change", SUBSYSTEM=="usb|tty", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="ace2", MODE="0666"
```

To activate the rules, type
```console
$ sudo udevadm control --reload
```
and then disconnect and reconnect the devices.

For more details on how to set up `udev` rules, see the corresponding section
in the [OpenTitan documentation](https://docs.opentitan.org/doc/ug/install_instructions/#xilinx-vivado).


### Configuring the Setup

The main configuration of the OpenTitan SCA analysis setup is stored in the file
```
cw/cw305/capture.yaml
```
For example, this file allows to specify the FPGA bitstream to be loaded and
the OpenTitan application binary to execute. By default, the prebuilt binaries
delivered with this repository are used. If you want to use custom binaries,
open the file and edit the specified file paths.

## Performing Example SCA Attack on AES

SCA attacks are performed in two steps. First, the target device is operated
and power traces are capture. Second, the power traces are analyzed. This is
commonly referred to as the actual SCA attack as performing a different attack
on the same peripheral does not necessarily require the collection of new power
traces.

### Capture Power Traces

Make sure all boards and adapters are powered up and connected to your PC and
that you have adjusted the configuration in `cw/cw305/capture.yaml`
according to your system.

Then run the following commands:

```console
$ cd cw/cw305
$ ./simple_capture_traces.py --num-traces 5000 --plot-traces 5000
```
This script will load the OpenTitan FPGA bitstream to the target board, load
and start the application binary to the target via SPI, and then feed data in
and out of the target while capturing power traces on the capture board. It
should produce console output similar to the following output:

```console
Connecting and loading FPGA... Done!
Initializing PLL1
Programming OpenTitan with "objs/aes_serial_fpga_nexysvideo.bin"...
frame: 0x00000000 to offset: 0x00000000
frame: 0x00000001 to offset: 0x000003d8
frame: 0x00000002 to offset: 0x000007b0
frame: 0x00000003 to offset: 0x00000b88
frame: 0x00000004 to offset: 0x00000f60
frame: 0x00000005 to offset: 0x00001338
frame: 0x00000006 to offset: 0x00001710
frame: 0x00000007 to offset: 0x00001ae8
frame: 0x00000008 to offset: 0x00001ec0
frame: 0x00000009 to offset: 0x00002298
frame: 0x0000000A to offset: 0x00002670
frame: 0x8000000B to offset: 0x00002a48
Serial baud rate = 38400
Serial baud rate = 115200
Scope setup with sampling rate 100003589.0 S/s
Target simpleserial version: z01 (attempts: 2).
Using key: b'2b7e151628aed2a6abf7158809cf4f3c'
Reading from FPGA using simpleserial protocol.
Checking version: 
Capturing: 100%|████████████████████████████| 5000/5000 [00:57<00:00, 86.78it/s]
```

In case you see console output like
```console
WARNING:root:Your firmware is outdated - latest is 0.20. Suggested to update firmware, as you may experience errors
See https://chipwhisperer.readthedocs.io/en/latest/api.html#firmware-update
```
you should update the firmware of one or both of the ChipWhisperer boards.
This process is straightforward and well documented online. Simply follow
the link printed on the console.

Once the power traces have been collected, a picture similar to the following
should be shown in your browser.

![](sample_traces.png)

Sample analysis run (Failed):

### Perform the Attack

To perform the attack, run the following command:

```console
$ ./simple_cpa_attack.py
```

This should produce console output similar to the output below in case
the attack fails:

```console
Performing Attack: 100%|████████████████████| 5000/5000 [03:41<00:00, 22.77it/s]
known_key: b'2b7e151628aed2a6abf7158809cf4f3c'
key guess: b'97f0e4c3bc14ff64effa5fce0697d9e0'
Subkey KGuess Correlation
 00    0x84    0.05677
 01    0xC1    0.05662
 02    0xF1    0.06032
 03    0x5F    0.05537
 04    0x7E    0.06062
 05    0xBF    0.05854
 06    0x6B    0.06673
 07    0x84    0.06018
 08    0x01    0.05967
 09    0x73    0.06864
 10    0xD7    0.05885
 11    0x92    0.05246
 12    0x18    0.06996
 13    0xA1    0.05271
 14    0x62    0.05984
 15    0xB2    0.05733

FAILED: key_guess != known_key
        0/16 bytes guessed correctly.
Saving results

```

Note that this particular attack is supposed to fail on the OpenTitan AES
implementation as the attack tries to exploit the Hamming distance leakage
of the state register in the last round, whereas the hardware does not write
the output of the last round back into the state register.
