# Why do we need a C-Extension?
At the moment, there are no Python implementations of [NIST SP 8000-185](https://doi.org/10.6028/NIST.SP.800-185).
However, there is a [feature request](https://bugs.python.org/issue39539) to add this functionality to [hashlib](https://docs.python.org/3/library/hashlib.html).
Thus, this extension might become obsolete once this feature is implemented as part of hashlib.
In the meantime, pyXKCP can be used to verify the KMAC hardware implementation.

## pyXKCP
pyXKCP is a python wrapper for the [eXtended Keccak Code Package (XKCP)](https://github.com/XKCP/XKCP).
XKCP features different high- and lowlevel implementation of Keccak.
At the moment only the highlevel function KMAC and the lowlevel implementation `compact` are supported.

### Building pyXKCP
There shouldn't be anything to do for the user.
The extension is build on-the-fly if not existing.
To explicitly build the SP800-185-C-extension `pyxkcp_build.py` can be used.
This will generate the needed configuration, compile the sources, and provides a shared library.

### Example
To see an example call `pyxkcp.test()`.

### License
Most of the used source files from XKCP are released to the **public domain** and associated to the [CC0](http://creativecommons.org/publicdomain/zero/1.0/) deed.
The only exception is the following:

* [`brg_endian.h`](../vendor/xkcp_xkcp/brg_endian.h) is copyrighted by Brian Gladman and comes with a BSD 3-clause license;
