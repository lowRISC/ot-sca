#!/bin/bash
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Check if CECA script can process CW or OT trace database.
# Only use 100 traces for CECA attack.
# Expected output is: "Failed to recover the AES key"
cmds=(
    '../analysis/ceca.py -f ci_data/ceca/aes_cw305_cw_db.cwp -d output -s 3 -w 4 -a 1060 1119 -n 100'
    '../analysis/ceca.py -f ci_data/ceca/aes_cw305_ot_db.db -d output -s 3 -w 4 -a 1060 1119 -n 100'
)
cmd_identifiers=(
    'CW'
    'OT trace'
)
for i in "${!cmds[@]}"; do
    OUT=$(${cmds[$i]} 2>&1)
    RETURN=$?

    if [[ "$RETURN" -eq 1 ]]; then
        if [[ $OUT == *"Failed to recover the AES key"* ]]; then
            echo "CECA "${cmd_identifiers[i]}" database successful!"
        else
        echo "${OUT}"
        echo "CECA "${cmd_identifiers[i]}" database unexpected output!"
            exit 1
        fi
    else
        echo "${OUT}"
        echo "CECA "${cmd_identifiers[i]}" database wrong error code!"
        exit 1
    fi
done

exit 0
