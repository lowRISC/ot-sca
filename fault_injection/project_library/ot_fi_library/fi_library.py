# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import sqlalchemy as db
import sqlalchemy_utils as db_utils
from sqlalchemy.orm import sessionmaker


@dataclass
class Metadata:
    """ Metadata for capture.

    Stores information about the FI experiment.

    """
    data: bytearray


@dataclass
class FIResult:
    """ FIResult.

    Defines FI result database entry.

    """
    fi_id: Optional[int] = 0
    response: Optional[str] = ""
    fi_result: Optional[int] = 0
    trigger_delay: Optional[float] = 0
    glitch_voltage: Optional[float] = 0
    glitch_width: Optional[float] = 0
    x_pos: Optional[int] = 0  # X position of x-y table
    y_pos: Optional[int] = 0  # Y position of x-y table
    # TODO: add entries for LFI, EMFI, and clock glitching


class FILibrary:
    """ Class for writing/reading FIResults to a database.

    Either a new database is created or FIResults are appended to the existing
    one. FIResults are first written into memory and then flushed into the
    database after reaching a FIResults memory threshold.
    """
    def __init__(self, db_name, fi_threshold, overwrite = False):
        if overwrite:
            fi_lib_file = Path(db_name + ".db")
            fi_lib_file.unlink(missing_ok=True)
        self.engine = db.create_engine("sqlite:///" + db_name + ".db")
        db_utils.create_database(self.engine.url)
        self.session = sessionmaker(self.engine)()
        self.metadata = db.MetaData()
        self.firesults_table = db.Table(
            "firesults",
            self.metadata,
            db.Column("fi_id", db.Integer, primary_key=True,
                      autoincrement=True),
            db.Column("response", db.String),
            db.Column("fi_result", db.Integer),
            db.Column("trigger_delay", db.Float),
            db.Column("glitch_voltage", db.Float),
            db.Column("glitch_width", db.Float),
            db.Column("x_pos", db.Integer),
            db.Column("y_pos", db.Integer),
        )
        self.metadata_table = db.Table(
            "metadata",
            self.metadata,
            db.Column("data", db.PickleType)
        )
        self.metadata.create_all(self.engine)
        self.fi_mem = []
        self.fi_mem_thr = fi_threshold

    def flush_to_disk(self):
        """ Writes FIResults from memory into database.
        """
        if self.fi_mem:
            query = db.insert(self.firesults_table)
            traces = []
            for trace in self.fi_mem:
                tr = asdict(trace)
                del tr["fi_id"]
                traces.append(tr)
            self.session.execute(query, traces)
            self.session.commit()
            self.fi_mem = []

    def write_to_buffer(self, fi_result):
        """ Write FIResults into memory or into storage.

        The FIResult is first written into memory. When the length of the
        buffer reaches the fi_mem threshold, the FIResults are flushed into the
        database storage.

        Args:
            trace: The trace to write.
        """
        self.fi_mem.append(fi_result)
        if len(self.fi_mem) >= self.fi_mem_thr:
            self.flush_to_disk()

    def get_firesults(self, start: Optional[int] = None,
                      end: Optional[int] = None):
        """ Get FIResults from database and store into RAM.

        Fetch FIResults from start to end from database storage into RAM.

        Args:
            start: fetch FIResults from index start
            end: to index end. If no start and end is provided, all traces are
                returned.
        Returns:
            The FIResults from the database.
        """
        self.flush_to_disk()
        if (start is not None) and (end is not None):
            # SQL ID starts at 1.
            start = start + 1
            query = db.select(self.firesults_table).where(
                (self.firesults_table.c.fi_id >= start) &
                (self.firesults_table.c.fi_id <= end))
        elif (start is not None):
            # SQL ID starts at 1.
            start = start + 1
            query = db.select(self.firesults_table).where(
                (self.firesults_table.c.fi_id == start))
        else:
            query = db.select(self.firesults_table)

        return [FIResult(**firesult._mapping)
                for firesult in self.session.execute(query).fetchall()]

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
