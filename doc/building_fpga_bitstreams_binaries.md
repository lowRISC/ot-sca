# Building FPGA Bitstreams & Binaries for OT-SCA

The purpose of this tutorial is to document how to build FPGA bitstreams and binaries that can be used for performing side-channel analysis and faults attacks on the ChipWhisperer boards using ot-sca.

## Prerequisites

Bitstreams and binaries are created inside the [OpenTitan](https://github.com/lowRISC/OpenTitan) repository.
Make sure to follow the getting started [guide](https://opentitan.org/book/doc/getting_started/index.html) before attempting to build bitstreams.
A Xilinx Vivado installation with a valid license is needed.

## Building CW310 Bitstreams

Depending on the evaluation goal (e.g., performing SCA evaluation of AES) different bitstreams are required.
The reason for this is that each bitstream is tailored to the needs of the evaluation goal (e.g., a hardware trigger is enabled for AES).

The following table gives an overview of the different bitstreams:

| Evaluation Goal | Bitstream Modifications                                                                                                                                                                                                                                                  | Branch Name  | Bitstream Name                                             |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------------------------------------------|
| AES SCA         | <ul><li>AES RTL is modified to raise a hardware-based trigger signal for SCA measurements</li><li>Frequency is reduced to 10MHz</li><li>HMAC, OTBN, SPI is stubbed and flash ECC and scrambling is disabled to reduce resource utilization</li></ul>                     | ot-sca_cw310_aes   | lowrisc_systems_chip_earlgrey_cw310_0.1_aes.bit  |
| KMAC SCA        | <ul><li>KMAC DOM masking is enabled and a hardware-based trigger signal for SCA measurements is enabled</li><li>Frequency is reduced to 10MHz</li><li>AES, HMAC, OTBN, SPI is stubbed and flash ECC and scrambling is disabled to reduce resource utilization</li></ul>  | ot-sca_cw310_kmac  | lowrisc_systems_chip_earlgrey_cw310_0.1_kmac.bit |
| OTBN SCA        | <ul><li>Frequency is reduced to 10MHz</li><li>AES, HMAC, SPI is stubbed and flash ECC and scrambling is disabled to reduce resource utilization</li></ul> | ot-sca_cw310_otbn  | lowrisc_systems_chip_earlgrey_cw310_0.1_otbn.bit |
| HMAC SCA        | No modifications, default bitstream                                                                                                                                                                                                                                      | ot-sca_cw310       | lowrisc_systems_chip_earlgrey_cw310_0.1.bit      |
| SHA3 SCA        | <ul><li>KMAC DOM masking is enabled and a hardware-based trigger signal for SCA measurements is enabled</li><li>Frequency is reduced to 10MHz</li><li>AES, HMAC, OTBN, SPI is stubbed and flash ECC and scrambling is disabled to reduce resource utilization</li></ul>  | ot-sca_cw310_kmac  | lowrisc_systems_chip_earlgrey_cw310_0.1_kmac.bit |
| Ibex SCA        | No modifications, default bitstream                                                                                                                                                                                                                                      | ot-sca_cw310       | lowrisc_systems_chip_earlgrey_cw310_0.1.bit      |
| FI              | No modifications, default bitstream                                                                                                                                                                                                                                      | ot-sca_cw310       | lowrisc_systems_chip_earlgrey_cw310_0.1.bit      |
| FI OTBN         | No modifications, default bitstream                                                                                                                                                                                                                                      | ot-sca_cw310       | lowrisc_systems_chip_earlgrey_cw310_0.1.bit      |

### Build Instructions

To build one of the bitstreams in the table above, perform the following:

```console
git clone https://github.com/lowRISC/opentitan.git && cd opentitan
git checkout <Branch Name>
./bazelisk.sh build //hw/bitstream/vivado:fpga_cw310_test_rom
```

The bitstream is located in
```console
bazel-bin/hw/bitstream/vivado/build.fpga_cw310/synth-vivado/lowrisc_systems_chip_earlgrey_cw310_0.1.bit
```
and can be copied into the `objs/` directory of ot-sca.

## Building CW310 Binaries

A binary is the compiled software penetration testing framework that is executed on the FPGA.
Similar to the bitstreams, for each evalution target, a different binary is required.

The following table gives an overview of the different binary configurations:

| Evaluation Goal | Branch Name        | Binary Name                   | Compilation Target                       |
|-----------------|--------------------|-------------------------------|------------------------------------------|
| AES SCA         | ot-sca_cw310_aes   | sca_aes_ujson_fpga_cw310.bin  | fpga_pentest_sca_fpga_cw310_test_rom     |
| KMAC SCA        | ot-sca_cw310_kmac  | sca_kmac_ujson_fpga_cw310.bin | fpga_pentest_sca_fpga_cw310_test_rom     |
| OTBN SCA        | ot-sca_cw310_otbn  | sca_otbn_ujson_fpga_cw310.bin | fpga_pentest_sca_fpga_cw310_test_rom     |
| HMAC SCA        | ot-sca_cw310       | sca_ujson_fpga_cw310.bin      | fpga_pentest_sca_fpga_cw310_test_rom     |
| SHA3 SCA        | ot-sca_cw310_kmac  | sca_kmac_ujson_fpga_cw310.bin | fpga_pentest_sca_fpga_cw310_test_rom     |
| Ibex SCA        | ot-sca_cw310       | sca_ujson_fpga_cw310.bin      | fpga_pentest_sca_fpga_cw310_test_rom     |
| FI              | ot-sca_cw310       | fi_ujson_fpga_cw310.bin       | fpga_pentest_fi_fpga_cw310_test_rom      |
| FI OTBN         | ot-sca_cw310       | fi_otbn_ujson_fpga_cw310.bin  | fpga_pentest_fi_otbn_fpga_cw310_test_rom |

### Build Instructions

To build one of the binaries in the table above, perform the following:

```console
git clone https://github.com/lowRISC/opentitan.git && cd opentitan
git checkout <Branch Name>
./bazelisk.sh build //sw/device/tests/penetrationtests/firmware:<Compilation Target>
```

The binary is located in
```console
bazel-bin/sw/device/tests/penetrationtests/firmware/<Compilation Target>.bin
```
and can be copied into the `objs/` directory of ot-sca.

## Pushing Bitstreams & Binaries to OT-SCA

When pushing bitstream and binaries to the ot-sca repository, please make sure:
- To include the git commit hash used for building the bitstream & binary in the commit message.
- That the bitstream & binary is tracked in git lfs and is not pushed in plain.