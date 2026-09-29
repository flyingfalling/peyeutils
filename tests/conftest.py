"""Shared pytest configuration.

Makes ``import peyeutils`` work whether or not the package has been
``pip install``-ed, by falling back to the ``src`` layout used in this repo.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
