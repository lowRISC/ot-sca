"""Class of tests for OpenTitan FI framework.
"""


class OTFITest:
    def __init__(self, name, cmd=None, mode=None):
        self.name = name
        self.cmd = "".join(s.capitalize() for s in name.split("_")) if cmd is None else cmd
        self.mode = mode
