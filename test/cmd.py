# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to run command-line programs and interpret and validate their results."""

import subprocess
from typing import Union


class Args:
    """Command-line arguments"""

    def __init__(self, cmd: Union[str, "list[str]"]):
        """Construct arguments from a string or from a list of strings.

        If a single string is given, split the string on spaces to form separate arguments.  If a
        list of strings is given, take each string in the list as one argument.
        """
        if isinstance(cmd, str):
            self._args = cmd.split(" ")
        else:
            self._args = cmd

    def __add__(self, other: "Args") -> "Args":
        """Concatenate two argument lists and return the result."""
        return Args(list(self) + list(other))

    def __getitem__(self, index: int) -> str:
        """Get the argument at index."""
        return self._args[index]

    def __iter__(self):
        """Iterate over the arguments."""
        for arg in self._args:
            yield arg

    def __repr__(self) -> str:
        """Return a printable representation of the Args object."""
        return f"Args({list(self)})"

    def __setitem__(self, index: int, value: str) -> None:
        """Set the argument at index to value."""
        self._args[index] = value


class Cmd:
    """A command that can be run as subprocess"""

    def __init__(self, args: Args):
        """Construct a command from arguments and set its expected returncode to 0."""
        self._args = args
        self._proc = None
        self._exp_returncode = 0

    def __repr__(self) -> str:
        return f"Cmd({self._args})"

    def run(self) -> "Cmd":
        """Run the command as subprocess and capture its stdout and stderr.

        If the expected returncode is not None, assert that it matches the actual returncode.
        """
        self._proc = subprocess.Popen(
            self._args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._stdout, self._stderr = self._proc.communicate()
        self._returncode = self._proc.returncode

        if self._exp_returncode is not None:
            assert self._returncode == self._exp_returncode, (
                f"{self._args} returned {self._returncode} instead of {self._exp_returncode}, "
                f"with the following stderr:\n{self.stderr_utf8()}"
            )

        return self

    def set_exp_returncode(self, exp_returncode) -> "Cmd":
        """Set the expected returncode.

        To disable any assertions on the returncode, set the expected returncode to None.
        """
        self._exp_returncode = exp_returncode
        return self

    def stderr(self) -> bytes:
        """Return the standard error of a command that has been run."""
        return self._stderr

    def stderr_utf8(self) -> str:
        """Return the standard error of a command that has been run, decoded as UTF-8 string."""
        return self.stderr().decode("utf-8")

    def stdout(self) -> bytes:
        """Return the standard output of a command that has been run."""
        return self._stdout

    def stdout_utf8(self) -> str:
        """Return the standard output of a command that has been run, decoded as UTF-8 string."""
        return self.stdout().decode("utf-8")
