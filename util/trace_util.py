# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def check_range(waves, bits_per_sample):
    """ The ADC output is in the interval [0, 2**bits_per_sample-1]. Check that the recorded
        traces are within [1, 2**bits_per_sample-2] to ensure the ADC doesn't saturate. """
    adc_range = np.array([0, 2**bits_per_sample])
    if not (np.all(np.greater(waves[:], adc_range[0])) and
            np.all(np.less(waves[:], adc_range[1] - 1))):
        print('\nWARNING: Some samples are outside the range [' +
              str(adc_range[0] + 1) + ', ' + str(adc_range[1] - 2) + '].')
        print('The ADC has a max range of [' +
              str(adc_range[0]) + ', ' + str(adc_range[1] - 1) + '] and might saturate.')
        print('It is recommended to reduce the scope gain.')
