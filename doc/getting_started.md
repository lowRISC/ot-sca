## Getting Started

TODO: Add ChipWhisperer and OpenTitan getting started instructions.

In the meantime, check the following PRs:

* https://github.com/lowRISC/opentitan/pull/2587
* https://github.com/lowRISC/opentitan/pull/2735

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
Performing Attack: 100%|████████████████████| 5000/5000 [02:31<00:00, 33.21it/s]
known_key: b'2b7e151628aed2a6abf7158809cf4f3c'
key guess: b'd836a6637fe0c056e2ac7cbd7673a4ce'
Subkey KGuess Correlation
  00    0xD8    0.05796
  01    0x36    0.06194
  02    0xA6    0.05526
  03    0x63    0.05977
  04    0x7F    0.05814
  05    0xE0    0.05497
  06    0xC0    0.05825
  07    0x56    0.05971
  08    0xE2    0.06239
  09    0xAC    0.05983
  10    0x7C    0.05809
  11    0xBD    0.06266
  12    0x76    0.06423
  13    0x73    0.06197
  14    0xA4    0.06361
  15    0xCE    0.05945

FAIL: key_guess != known_key
Saving results
```
