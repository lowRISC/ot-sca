#!/bin/bash

# Lint all Python sources using the specified lint tools
FILE_PATHS=`find . -not \( -path './.venv' -prune \) \
                   -not \( -path './util/vendor' -prune \) \
                   -name '*.py'`

./util/vendor/lowrisc_opentitan/lintpy.py -f ${FILE_PATHS} --tools flake8,isort
