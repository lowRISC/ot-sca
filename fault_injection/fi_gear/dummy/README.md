# Template drivers for FI gear

This directory includes dummy drivers for FI gear that can be used as a
template to create new drivers.
All drivers need to implement these functions:
- init:
    - Initializes the FI gear.
- generate_fi_parameters:
    - Generate parameters for the FI parameter sweep. This can happen either
      randomly or deterministically.
- arm_trigger:
    - Configure the FI gear with the current FI parameters and arm the trigger.
- reset:
    - After a power cycle of the DUT, some FI gear also needs to be reset.
