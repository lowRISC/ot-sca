# Building FPGA Bitstreams for OT-SCA

The idea of this tutorial is to document how to build FPGA bitstreams that can be used for performing side-channel analysis on the ChipWhisperer boards using ot-sca.

## Prerequisites

Bitstreams are created inside the [OpenTitan](https://github.com/lowRISC/OpenTitan) repository.
Make sure to follow the getting started [guide](https://opentitan.org/book/doc/getting_started/index.html) before attempting to build bitstreams.
A Xilinx Vivado installation with a valid license is needed.

Then, clone the OpenTitan repository and select a suitable commit. Make sure that the selected commit is compatible with the penetration testing [framework](https://github.com/lowRISC/opentitan/tree/master/sw/device/tests/penetrationtests).
When using the latest commit available in the OpenTitan master branch, the framework and the FPGA bitstream should be compatible. 


## Building CW310 Bitstreams

For SCA measurements on the CW310, different bitstreams are available.

| Name                                                   | Description                                                                                              |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| lowrisc_systems_chip_earlgrey_cw310_0.1_X.bit          | Default bitstream. AES RTL is modified to raise a hardware-based trigger signal for SCA measurements.    |
| lowrisc_systems_chip_earlgrey_cw310_0.1_kmac_dom_X.bit | Enables DOM masking for KMAC. Removes some blocks (e.g., OTBN) to save resource utilization on the FPGA. |
| lowrisc_systems_chip_earlgrey_cw310_0.1_ecdsa_X.bit    |                                                                                                          |

The table above highlights the difference between each bitstream.
`X` denotes the hardware version, e.g., `Z1` or `A1`.

### Building the Default CW310 Bitstream

To build the default bitstream, change the following parameters marked in bold in the [chip_earlgrey_cw310.sv](https://github.com/lowRISC/opentitan/blob/master/hw/top_earlgrey/rtl/autogen/chip_earlgrey_cw310.sv) file:

<pre>
  top_earlgrey #(
    .SecAesMasking(1'b1),
    .SecAesSBoxImpl(aes_pkg::SBoxImplDom),
    .SecAesStartTriggerDelay(<b>320</b>),
    .SecAesAllowForcingMasks(1'b1),
    .CsrngSBoxImpl(aes_pkg::SBoxImplLut),
    .OtbnRegFile(otbn_pkg::RegFileFPGA),
    .SecOtbnMuteUrnd(<b>1'b1</b>),
    .SecOtbnSkipUrndReseedAtStart(<b>1'b0</b>),
    .OtpCtrlMemInitFile(OtpCtrlMemInitFile),
    .RvCoreIbexPipeLine(1),
    .SramCtrlRetAonInstrExec(0),
    .UsbdevRcvrWakeTimeUs(10000),
    .KmacEnMasking(0),
    .KmacSwKeyMasked(1),
    .KeymgrKmacEnMasking(0),
    .SecKmacCmdDelay(<b>320</b>),
    .SecKmacIdleAcceptSwMsg(<b>1'b1</b>),
    .RomCtrlBootRomInitFile(BootRomInitFile),
    .RvCoreIbexRegFile(ibex_pkg::RegFileFPGA),
    .RvCoreIbexSecureIbex(0),
    .SramCtrlMainInstrExec(1),
    .PinmuxAonTargetCfg(PinmuxTargetCfg)
```
</pre>

With these hardware changes, the AES and KMAC IP block delay the start of the operation by 320 cycles, giving Ibex enough time to go into the sleep mode.
Moreover, the KMAC IP block is configured to accept writes to the KMAC message FIFO before triggering the operations.
Finally, OTBN is configured in such a way that PRNG reseeds are skipped at the OTBN application start.

After making these  hardware changes, use the following command to retrieve the bitstream:

```console
./bazelisk.sh build //hw/bitstream/vivado:fpga_cw310_test_rom
```

The bitstream is located in
```console
bazel-bin/hw/bitstream/vivado/build.fpga_cw310/synth-vivado/lowrisc_systems_chip_earlgrey_cw310_0.1.bit
```
and can be copied into the `objs/` directory of ot-sca.

### Building the KMAC DOM CW310 Bitstream

### Building the ECDSA CW310 Bitstream

## Building CW305 Bitstreams

#TODO(lowrisc/ot-sca#377): Add instructions how to build bitstreams for CW305.

## Pushing Bitstreams to OT-SCA

When pushing bitstream to the ot-sca repository, please make sure:
- To include the git commit hash used for building the bitstream in the commit message.
- That the bitstream is compatible with the penetration testing firmware.
- That the bitstream is tracked in git lfs and is not pushed in plain.
