# Penetrationtests

## Overview
This directory contains the host and test scripts for the fault injection and side-channel tests from OpenTitan's penetrationtest OS.  
More specifically, the host scripts contains examples on how to combine the commands to the penetrationtest OS from //target/communication in order to perform simple tests such as running an AES encryption.  
The test scripts then provide information on how to verify the responds from each test such as their gold response or providing info how the batching is performed. These test scripts are also run from the OpenTitan repo to ensure all scripts are functioning correctly.

## How to run tests
Conceptually, the Bazel framework from the OpenTitan repo is extended by the ot-sca repo. Hence, ot-sca's build targets are external build targets to OpenTitan.
In order to run tests, add the root of the ot-sca repo as follows
```
export OTSCA_EXTS_DIR=”path/to/ot-sca/”
```
You can also place the above in your ~/.bashrc file to not redo this step every new kernel.

From the OpenTitan repository, it is possible to run a test such as with the following command
```
./bazelisk.sh test  @otsca_exts//test/penetrationtests/sca:sca_hmac_test --define=FW_ENV=silicon_owner_sival_rom_ext
```

The define FW_ENV flag is used to decide which target to build. Examples of FW_ENV can be silicon_owner_gb_rom_ext or fpga_cw310_sival_rom_ext, etc.  
The @otsca_exts word is used to denote that the target is in the external repo.  
This command will build the corresponding binary in //sw/device/tests/penetrationtests and then execute the test from @otsca_exts//test/penetrationtests/.
