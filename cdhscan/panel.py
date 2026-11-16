"""The panel reference: clean panel median and MAD of every catalogue locus.

A person's model input at a locus is their own values, own minus the clean panel median, and the
clean panel MAD. The clean panel of a locus is every panel person who does not carry a CDH there.
The released panel is the 171 training people. Per chromosome, one .npz holds, per locus (first
axis):
  locus_id, n_clean
  balanced_median, balanced_mad   balanced log2(O/E) of the 1 Mb window, upper triangle (5050 values)
  raw_median, raw_mad             raw log2(O/E) of the window, upper triangle
  coverage_median, coverage_mad   log relative coverage of the window's 100 bins
  wgs_median, wgs_mad             the 4 WGS values of the window's 100 bins (100 x 4)
"""

from pathlib import Path

import numpy as np

from .constants import WINDOW_BINS
from .features import as_map_channels

UPPER = np.triu_indices(WINDOW_BINS)
TRIANGLE_ARRAYS = ["balanced_median", "balanced_mad", "raw_median", "raw_mad"]
VECTOR_ARRAYS = ["coverage_median", "coverage_mad", "wgs_median", "wgs_mad"]


def symmetric(upper_values):
    matrix = np.zeros((WINDOW_BINS, WINDOW_BINS), dtype=np.float32)
    matrix[UPPER] = upper_values
    return matrix + np.triu(matrix, 1).T


class ChromosomePanel:
    """The panel reference of the loci of one chromosome."""

    def __init__(self, path):
        stored = np.load(path)
        self.arrays = {name: stored[name] for name in TRIANGLE_ARRAYS + VECTOR_ARRAYS}
        self.row = {locus_id: i for i, locus_id in enumerate(stored["locus_id"])}

    def __contains__(self, locus_id):
        return locus_id in self.row

    def map_statistics(self, locus_id):
        """Clean panel median and MAD in the 4 own-map layout (4 x 100 x 100 each)."""
        i = self.row[locus_id]
        median = as_map_channels(symmetric(self.arrays["balanced_median"][i]), symmetric(self.arrays["raw_median"][i]),
                                 self.arrays["coverage_median"][i])
        mad = as_map_channels(symmetric(self.arrays["balanced_mad"][i]), symmetric(self.arrays["raw_mad"][i]),
                              self.arrays["coverage_mad"][i])
        return median, mad

    def wgs_statistics(self, locus_id):
        """Clean panel median and MAD of the WGS values (100 x 4 each)."""
        i = self.row[locus_id]
        return self.arrays["wgs_median"][i], self.arrays["wgs_mad"][i]


def load_panel(panel_dir, chrom):
    return ChromosomePanel(Path(panel_dir) / f"{chrom}.npz")
