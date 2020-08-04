## Getting Started

TODO: Add ChipWhisperer and OpenTitan getting started instructions.

In the meantime, check the following PRs:

* https://github.com/lowRISC/opentitan/pull/2587
* https://github.com/lowRISC/opentitan/pull/2735

### Downloading code via FTDI

Currently the target firmware is downloaded via a SPI FTDI interface using the
[spiflash](https://docs.opentitan.org/sw/host/spiflash/README/) utility.
A debian-compatible pre-built binary is available at
`cw/cw305/bin/linux/spiflash`. See the following table for details on how to
connect the FTDI interface to the CW305 FPGA target board.

| FTDI Signal | CW305 FPGA | OpenTitan IO        |
| ----------- | ---------- | ------------------- |
| TCK         | JP3.A14    | TCK (IO_DPS0)       |
| TDI         | JP3.A13    | TDI (IO_DPS1)       |
| TDO         | JP3.A15    | TDO (IO_DPS2)       |
| TMS         | JP3.B15    | TMS (IO_DPS3)       |
| GPIOL0      | JP3.C12    | nTRST (IO_DPS4)     |
| GPIOL1      | JP3.C11    | nSRST (IO_DPS5)     |
| GPIOL2      | JP3.B14    | JTAG_SEL (IO_DPS6)  |
| GPIOL3      | JP3.C14    | BOOTSTRAP (IO_DPS7) |
| GND         | GND        | -                   |

## Running capture and attack

Sample capture run:

```console
$ cd cw/cw305
$ python3 capture_traces.py
Connecting and loading FPGA
Initializing PLL1
Running SPI flash update.
Image divided into 8 frames.
frame: 0x00000000 to offset: 0x00000000
frame: 0x00000001 to offset: 0x000003d8
frame: 0x00000002 to offset: 0x000007b0
frame: 0x00000003 to offset: 0x00000b88
frame: 0x00000004 to offset: 0x00000f60
frame: 0x00000005 to offset: 0x00001338
frame: 0x00000006 to offset: 0x00001710
frame: 0x80000007 to offset: 0x00001ae8
Serial baud rate = 38400
Serial baud rate = 57600
Using key: b'2b7e151628aed2a6abf7158809cf4f3c'
Reading from FPGA using simpleserial protocol.
Checking version:
Capturing: 100%|████████████████████████████| 5000/5000 [01:35<00:00, 52.26it/s]
Saving sample trace image to: doc/sample_traces.html
```

![](sample_traces.png)

Sample analysis run (Failed):

```console
$ python3 cpa_attack.py
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

FAIL: key_guess != known_key
Saving results

```
