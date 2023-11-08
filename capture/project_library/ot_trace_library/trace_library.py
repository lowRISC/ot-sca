# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sqlalchemy as db
import sqlalchemy_utils as db_utils
from sqlalchemy.orm import sessionmaker


@dataclass
class Metadata:
    """ Metadata for capture.

    Stores information about the capture of the traces.

    """
    data: bytearray


@dataclass
class Trace:
    """ Trace.

    Defines the trace format.

    """
    wave: bytearray
    plaintext: bytearray
    ciphertext: bytearray
    key: bytearray
    x_pos: Optional[int] = 0  # X position of probe
    y_pos: Optional[int] = 0  # Y position of probe
    trace_id: Optional[int] = 0


class TraceLibrary:
    """ Class for writing/reading traces to a database.

    Either a new database is created or traces are appended to the existing
    one. Traces are first written into memory and the flushed into the
    database after reaching a trace memory threshold.
    """
    def __init__(self, db_name, trace_threshold, wave_datatype = np.uint16,
                 overwrite = False):
        if overwrite:
            trace_lib_file = Path(db_name + ".db")
            trace_lib_file.unlink(missing_ok=True)
        self.engine = db.create_engine("sqlite:///" + db_name + ".db")
        db_utils.create_database(self.engine.url)
        self.session = sessionmaker(self.engine)()
        self.metadata = db.MetaData()
        self.traces_table = db.Table(
            "traces",
            self.metadata,
            db.Column("trace_id", db.Integer, primary_key=True,
                      autoincrement=True),
            db.Column("wave", db.LargeBinary),
            db.Column("plaintext", db.LargeBinary),
            db.Column("ciphertext", db.LargeBinary),
            db.Column("key", db.LargeBinary),
            db.Column("x_pos", db.Integer),
            db.Column("y_pos", db.Integer),
        )
        self.metadata_table = db.Table(
            "metadata",
            self.metadata,
            db.Column("data", db.PickleType)
        )
        self.metadata.create_all(self.engine)
        self.trace_mem = []
        self.trace_mem_thr = trace_threshold
        self.wave_datatype = wave_datatype

    def flush_to_disk(self):
        """ Writes traces from memory into database.
        """
        if self.trace_mem:
            query = db.insert(self.traces_table)
            traces = []
            for trace in self.trace_mem:
                tr = asdict(trace)
                del tr["trace_id"]
                traces.append(tr)
            self.session.execute(query, traces)
            self.session.commit()
            self.trace_mem = []

    def write_to_buffer(self, trace):
        """ Write traces into memory or into storage.

        The trace is first written into memory. When the length of the buffer
        reaches the memory threshold, the traces are flushed into the database
        storage.

        Args:
            trace: The trace to write.
        """
        self.trace_mem.append(trace)
        if len(self.trace_mem) >= self.trace_mem_thr:
            self.flush_to_disk()

    def get_traces(self, start: Optional[int] = None,
                   end: Optional[int] = None):
        """ Get traces from database and stored into RAM.

        Fetch traces from start to end from database storage into RAM.

        Args:
            start: fetch traces from trace index start
            end: to trace index end. If no start and end is provided, all
                 all traces are returned.
        Returns:
            The traces from the database.
        """
        self.flush_to_disk()
        if start and end:
            query = db.select(self.traces_table).where(
                (self.traces_table.c.trace_id >= start) &
                (self.traces_table.c.trace_id <= end))
        else:
            query = db.select(self.traces_table)

        return [Trace(**trace._mapping)
                for trace in self.session.execute(query).fetchall()]

    def get_waves_bytearray(self, start: Optional[int] = None,
                            end: Optional[int] = None):
        """ Get all waves from the database.

        Returns:
            The bytearray waves from the database.
        """
        return [trace.wave for trace in self.get_traces(start, end)]

    def get_waves(self, start: Optional[int] = None, end: Optional[int] = None):
        """ Get all waves from the database in the trace array format.

        Returns:
            The waves from the database in the type wave_datatype.
        """
        return [np.frombuffer(b, self.wave_datatype)
                for b in self.get_waves_bytearray(start, end)]

    def get_plaintexts(self, start: Optional[int] = None,
                       end: Optional[int] = None):
        """ Get all plaintexts between start and end from the database in the
        int8 array format.

        Returns:
            The int plaintexts from the database.
        """
        return [np.frombuffer(trace.plaintext, np.uint8)
                for trace in self.get_traces(start, end)]

    def get_keys(self, start: Optional[int] = None,
                 end: Optional[int] = None):
        """ Get all keys between start and end from the database in the int8
        array format.

        Returns:
            The int keys from the database.
        """
        return [np.frombuffer(trace.key, np.uint8)
                for trace in self.get_traces(start, end)]

    def write_metadata(self, metadata):
        """ Write metadata into database.

        Args:
           metadata: The metadata to store.
        """
        query = db.insert(self.metadata_table)
        data = Metadata(str(pickle.dumps(metadata), encoding="latin1"))
        self.session.execute(query, asdict(data))
        self.session.commit()

    def get_metadata(self):
        """ Get metadata from database.

        Returns:
            The metadata from the database.
        """
        query = db.select(self.metadata_table)
        metadata = Metadata(**self.session.execute(query).fetchall()[0]._mapping)
        return pickle.loads(bytes(metadata.data, encoding="latin1"))
