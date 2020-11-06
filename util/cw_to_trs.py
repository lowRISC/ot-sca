#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Module used to convert ChipWhisperer project files into Riscure's trs format.

Example:

./cw_to_trs.py --input=path/to/project_name --output=project_name.trs

The --export-key option can be used to include the test key in the traces, but
it is not required.
"""
import argparse
import binascii
import chipwhisperer as cw
import trsfile
from tqdm import tqdm


def gen_trs_headers(project, export_key):
  """Returns a trs file header with trace metadata information.

  Args:
    project: ChipWhisperer traces project.
    export_key: Set to True to include the size of the key in the LENGTH_DATA
        field.
  Returns:
    trs file header.
  """
  return {
    trsfile.Header.LABEL_X: 's',
    trsfile.Header.LABEL_Y: 'V',
    trsfile.Header.NUMBER_SAMPLES: len(project.waves[0]),
    trsfile.Header.TRACE_TITLE: 'title',
    trsfile.Header.TRACE_OVERLAP: False,
    trsfile.Header.GO_LAST_TRACE: False,
    trsfile.Header.SAMPLE_CODING: trsfile.SampleCoding.FLOAT,
    trsfile.Header.LENGTH_DATA: len(gen_trs_data(project.traces[0],
                                                 export_key)),

    # TODO: Hardcoded to 200mV. Consider calculating the range directly from
    # the traces.
    trsfile.Header.ACQUISITION_RANGE_OF_SCOPE: 0.200,
    trsfile.Header.ACQUISITION_COUPLING_OF_SCOPE: 1,
    trsfile.Header.ACQUISITION_OFFSET_OF_SCOPE: 0.0,
    trsfile.Header.ACQUISITION_DEVICE_ID: b'CWLite',
    trsfile.Header.ACQUISITION_TYPE_FILTER: 0,
  }


def calc_data_offsets(trace, export_key, header):
  """Calculate trs data offsets to textin, textout and key.

  Args:
    header: trs header. To be modified in place.
    export_key: Set to True to add KEY_OFFSET and KEY_LENGTH to the trs header.
    trace: ChipWhisperer trace.
  """
  input_offset = 0
  input_len = len(trace.textin)
  output_offset = input_offset + input_len
  output_len = len(trace.textout)

  key_offset = output_offset + output_len
  key_len = len(trace.key)

  header.update({
    trsfile.Header.INPUT_OFFSET: input_offset,
    trsfile.Header.INPUT_LENGTH: input_len,
    trsfile.Header.OUTPUT_OFFSET: output_offset,
    trsfile.Header.OUTPUT_LENGTH: output_len,
  })

  if export_key:
    header.update({
      trsfile.Header.KEY_OFFSET: key_offset,
      trsfile.Header.KEY_LENGTH: key_len,

    })


def gen_trs_data(trace, export_key):
  """Returns serialized trace textin, textout and key data in string format

  Args:
    trace: ChipWhisperer trace.
    export_key: Set to True to append the key to the end of the result string.
  Returns:
    Binary encoded concatenation of textin, textout and key.
  """
  data = bytearray(trace.textin) + bytearray(trace.textout)
  if export_key:
    return data + bytearray(trace.key)
  return data


def cw_project_to_trs(project_name, trs_filename, export_keys):
  """Converts ChipWhisperer project into trs trace format.

  Args:
    project_name: Path to ChipWhisperer capture project.
    trs_filename: Output filename for trs result.
    export_keys: Set to true to include the keys in the trs output.
  """
  print(f'input project: {project_name}')
  p = cw.open_project(project_name)
  print(f'num_traces: {len(p.traces)}')
  print(f'num_samples per trace: {len(p.waves[0])}')
  print(f'output file: {trs_filename}')

  h = gen_trs_headers(p, export_keys)
  calc_data_offsets(p.traces[0], export_keys, h)

  traces = []
  for trace in tqdm(p.traces, desc='Converting', ncols=80):
    traces.append(trsfile.Trace(trsfile.SampleCoding.FLOAT, trace.wave,
                                data=gen_trs_data(trace, export_keys)))

  print('Writing output file, this may take a while.')
  with trsfile.trs_open(trs_filename, 'w', engine='TrsEngine', headers=h,
                        live_update=True) as t:
    t.extend(traces)


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input',
                      '-i',
                      type=str,
                      required=True,
                      help="Input ChipWhisperer project.")
  parser.add_argument('--output',
                      '-o',
                      type=str,
                      required=True,
                      help="Output trs filename.")
  parser.add_argument('--export-key',
                      '-k',
                      action='store_true',
                      help="Include keys in data output.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = parse_args()
  cw_project_to_trs(args.input, args.output, args.export_key)
