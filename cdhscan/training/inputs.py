"""Model inputs of the training and validation windows.

Hi-C input of the window of person P at a locus (12 x 100 x 100, stored as float16):
  0-3   P's own map values (balanced log2(O/E), raw log2(O/E), row coverage, column coverage)
  4-7   own minus the clean panel median
  8-11  the clean panel MAD
WGS input (12 x 100): the same three parts for the 4 WGS values.
Clean panel: the panel people who do not carry a CDH at the locus, without P. A positive P is a
carrier and so never in it; a negative P is taken out of their own reference (leave-one-out), so
that negatives, like positives and like a new person, are never compared with themselves.
NaN values (blank in P's data or in the whole clean panel) become 0.
"""

import os
from pathlib import Path

import numpy as np

from ..constants import BIN_BP, MAP_CHANNELS, WGS_CHANNELS, WINDOW_BINS
from ..features import as_map_channels, median_and_mad, window_values


def clean_map_statistics(stacked_values, keep):
    """Median and MAD (4 x 100 x 100 each) over the clean people selected by keep."""
    medians, mads = zip(*(median_and_mad(values[keep]) for values in stacked_values))
    return as_map_channels(*medians), as_map_channels(*mads)


def write_hic_inputs(windows, carriers, panel, value_bands, out_file):
    """Hi-C inputs of the windows of one file (sorted by input_row), written to out_file.

    carriers: locus_id -> set of carriers; value_bands: person -> features.hic_value_bands."""
    temporary_file = out_file.with_name(out_file.name + ".tmp.npy")
    inputs = np.lib.format.open_memmap(temporary_file, mode="w+", dtype=np.float16,
                                       shape=(len(windows), len(MAP_CHANNELS), WINDOW_BINS, WINDOW_BINS))
    for locus_id, locus_windows in windows.groupby("locus_id", sort=False):
        first_bin = locus_windows["start"].iloc[0] // BIN_BP
        clean_people = [person for person in panel if person not in carriers[locus_id]]
        # Per value (balanced, raw, coverage): the clean people's window values stacked.
        stacked_values = [np.stack(part) for part in zip(*(window_values(value_bands[person], first_bin)
                                                           for person in clean_people))]
        full_statistics = None
        for i, window in locus_windows.iterrows():
            own = as_map_channels(*window_values(value_bands[window["sample"]], first_bin))
            if window["label"] == 1:
                if full_statistics is None:
                    full_statistics = clean_map_statistics(stacked_values, np.ones(len(clean_people), dtype=bool))
                median, mad = full_statistics
            else:
                median, mad = clean_map_statistics(stacked_values,
                                                   np.array([person != window["sample"] for person in clean_people]))
            inputs[i] = np.nan_to_num(np.concatenate([own, own - median, mad]), nan=0.0).astype(np.float16)
    inputs.flush()
    del inputs
    os.replace(temporary_file, out_file)


def wgs_tracks(windows, carriers, panel, values, first_rows):
    """WGS inputs (n_windows x 12 x 100) in windows order.

    values: person -> features.wgs_values; first_rows: chrom -> first row of the chromosome in bins.tsv."""
    tracks = np.empty((len(windows), len(WGS_CHANNELS), WINDOW_BINS), dtype=np.float32)
    for locus_id, locus_windows in windows.groupby("locus_id", sort=False):
        chrom, first_bin = locus_windows["chrom"].iloc[0], locus_windows["start"].iloc[0] // BIN_BP
        rows = first_rows[chrom] + first_bin + np.arange(WINDOW_BINS)
        clean_people = [person for person in panel if person not in carriers[locus_id]]
        clean_values = np.stack([values[person][rows] for person in clean_people])
        for i, window in locus_windows.iterrows():
            own = values[window["sample"]][rows]
            keep = np.array([person != window["sample"] for person in clean_people])
            median, mad = median_and_mad(clean_values[keep])
            tracks[i] = np.nan_to_num(np.concatenate([own, own - median, mad], axis=1), nan=0.0).T
    return tracks


def carriers_of(loci):
    return {row.locus_id: set(row.carriers.split(",")) for row in loci.itertuples()}
