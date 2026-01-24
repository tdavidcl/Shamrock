import json
import os

import numpy as np

import shamrock.sys

try:
    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from .ColumnDensityPlot import ColumnDensityPlot
from .PerfHistory import PerfHistory
from .SliceDensityPlot import SliceDensityPlot
from .SliceVzPlot import SliceVzPlot
from .StandardPlotHelper import StandardPlotHelper
from .UnitHelper import plot_codeu_to_unit
