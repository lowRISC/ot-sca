# OT-SCA Capture Scripts

The `caputure_*.py` scripts provided in this directory allow the user to
capture traces of a cryptographic operation performed by OpenTitan.

## Quick Usage

Follow the steps described in the [`getting_started`](../doc/getting_started.md)
documentation to setup the environment, the target, and the scope.

When using OpenTitan on the CW310 and measuring traces using Husky, AES traces
for a random key tests can be captured with the following command:
```console
mkdir -p projects/
./capture_aes.py --capture_config configs/aes_sca_cw310.yaml --project projects/aes_sca_cw310_random
```
The traces are then stored in the database file in `projects/`.

## Capture Config

The capture configs stored in `configs/` follow the following structure:
- target
- scope type (husky or waverunner)
- capture
- test

The target entry specifies the target. Currently, only the `cw310` or `cw305`
FGPA boards are supported. The scope entry defines (indirectly) the sampling
rate as well as the scope gain and cycle offset. With the capture entry, the
user can select the corresponding scope, configure the trace plot, and select
the trace database format (either chipwhisperer project or OT trace library in
the SQLite format). Test provides different entries for the cipher config.

## Adding new Cipher Capture Scripts

To add new cipher capture scripts, please use the `caputure_aes.py` script as
a template. The template follows the following structure:
- Setup the target, scope, and project
- Configure the cipher and establish the communication interface
- Capture traces
- Print traces

The communication interface (e.g., the individual simpleserial commands) for
each cipher needs to be added to `lib/ot_communication.py`.
