import os
with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from nics_fix_pt.quant import *
import nics_fix_pt.nn_fix_inner
