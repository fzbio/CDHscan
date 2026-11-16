"""Hi-C and WGS values of one person, and cutting them into 1 Mb windows.

Hi-C values on one chromosome, stored as bands (band[i, d] = value for bins i and i + d):
  balanced log2(O/E)  log2((contact + c) / (expected + c)); balanced contact = raw / (SCALE factor
                      of bin i x SCALE factor of bin j); expected = mean of the person's defined
                      balanced contacts at that distance on the chromosome; c = the smallest
                      positive balanced contact of the chromosome. NaN where the map is blank.
  raw log2(O/E)       log2((raw + 1) / (expected raw + 1)); expected raw = mean raw count at
                      that distance over bin pairs whose two bins both have reads
  log relative coverage of a bin  log2((coverage + 1) / (median coverage + 1)); coverage = read
                      pairs between the bin and any bin at most 99 bins away; median over the
                      chromosome's covered bins

WGS values per lifted bin, from the read counts:
  log_relative_depth      log2((depth_reads + 1) / (expected + 1)); expected = the person's median
                          depth_reads per GRCh38 bp over all lifted bins, times the bin's GRCh38
                          length (corrects for the person's overall sequencing depth)
  soft_clipped_fraction, mate_unmapped_fraction, discordant_fraction   among all_reads
A fraction is NaN in a bin without reads; every value is NaN in a bin that is not lifted.
"""

import warnings

import numpy as np

from .constants import COUNT_COLUMNS, WGS_VALUES, WINDOW_BINS

RAW_PSEUDOCOUNT = 1.0   # the smallest nonzero raw read-pair count


def bin_coverage(raw_band):
    """Read pairs between each bin and any bin at most 99 bins away, in either direction."""
    filled = np.nan_to_num(raw_band, nan=0.0)
    coverage = filled.sum(axis=1)
    for distance in range(1, WINDOW_BINS):
        coverage[distance:] += filled[:-distance, distance]
    return coverage


def raw_expected_by_distance(raw_band, coverage):
    """Mean raw count at each distance over the bin pairs whose two bins both have coverage."""
    n_bins = len(raw_band)
    has_coverage = coverage > 0
    expected = np.zeros(WINDOW_BINS)
    for distance in range(WINDOW_BINS):
        rows = np.arange(n_bins - distance)
        both_covered = has_coverage[rows] & has_coverage[rows + distance]
        expected[distance] = raw_band[rows[both_covered], distance].mean()
    return expected


def balanced_band(raw_band, scale_factor):
    """raw / (SCALE factor of bin i x SCALE factor of bin i + d); NaN where either bin has no factor."""
    n_bins = len(raw_band)
    partner = np.minimum(np.arange(n_bins)[:, None] + np.arange(WINDOW_BINS)[None, :], n_bins - 1)
    return raw_band / (scale_factor[:, None] * scale_factor[partner])


def hic_value_bands(raw_band, scale_factor):
    """One person's Hi-C value bands on one chromosome."""
    band = balanced_band(raw_band, scale_factor)
    expected = np.array([np.nanmean(band[:, distance]) for distance in range(WINDOW_BINS)])
    pseudocount = float(band[band > 0].min())
    coverage = bin_coverage(raw_band)
    median_coverage = np.median(coverage[coverage > 0])
    raw_expected = raw_expected_by_distance(raw_band, coverage)
    return {"balanced_log_oe": np.log2((band + pseudocount) / (expected + pseudocount)).astype(np.float32),
            "raw_log_oe": np.log2((raw_band + RAW_PSEUDOCOUNT) / (raw_expected + RAW_PSEUDOCOUNT)).astype(np.float32),
            "log_relative_coverage": np.log2((coverage + 1) / (median_coverage + 1)).astype(np.float32)}


def band_window(band, first_bin):
    """100 x 100 symmetric matrix of the window starting at first_bin, cut from a band."""
    index = np.arange(WINDOW_BINS)
    distance = np.abs(np.subtract.outer(index, index))
    rows = first_bin + np.minimum.outer(index, index)
    return band[rows, distance]


def window_values(value_bands, first_bin):
    """balanced log2(O/E) and raw log2(O/E) (100 x 100) and log relative coverage (100) of one window."""
    return (band_window(value_bands["balanced_log_oe"], first_bin),
            band_window(value_bands["raw_log_oe"], first_bin),
            value_bands["log_relative_coverage"][first_bin:first_bin + WINDOW_BINS])


def as_map_channels(balanced, raw, coverage):
    """The 4 own-map layout: balanced, raw, row coverage (entry i, j = coverage of bin i), column coverage."""
    row_matrix = np.repeat(coverage[:, None], WINDOW_BINS, axis=1)
    return np.stack([balanced, raw, row_matrix, row_matrix.T])


def wgs_values(counts, bins):
    """The person's WGS values (n_bins x 4, WGS_VALUES order) from the read counts (n_bins x 8)."""
    column = {name: counts[:, COUNT_COLUMNS.index(name)].astype(np.float64) for name in COUNT_COLUMNS}
    lifted = bins["lifted"].to_numpy()
    grch38_bp = (bins["grch38_end"] - bins["grch38_start"]).to_numpy().astype(np.float64)
    reads_per_bp = np.median(column["depth_reads"][lifted] / grch38_bp[lifted])
    expected = reads_per_bp * grch38_bp
    values = np.full((len(bins), len(WGS_VALUES)), np.nan)
    values[:, 0] = np.log2((column["depth_reads"] + 1) / (expected + 1))
    has_reads = column["all_reads"] > 0
    for i, name in enumerate(["soft_clipped", "mate_unmapped", "discordant"], start=1):
        values[has_reads, i] = column[name][has_reads] / column["all_reads"][has_reads]
    values[~lifted] = np.nan
    return values.astype(np.float32)


def median_and_mad(values):
    """Median and MAD over axis 0 (people), ignoring NaN; NaN where every person is NaN."""
    with warnings.catch_warnings():
        # A value that is blank in every person gives NaN, which is the intended result.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median), axis=0)
    return median.astype(np.float32), mad.astype(np.float32)


def blank_bins_per_window(valid_bin, first_bins):
    """Bins without a SCALE factor in each window (prefix sums of the blank-bin indicator)."""
    blank_total = np.concatenate([[0], np.cumsum(~valid_bin)])
    return blank_total[first_bins + WINDOW_BINS] - blank_total[first_bins]


def locus_mask(locus_start, locus_end, window_start, bin_bp):
    """Boolean per window bin: does the bin lie in the locus?"""
    index = np.arange(WINDOW_BINS)
    first = np.clip((locus_start - window_start) // bin_bp, 0, WINDOW_BINS - 1)
    last = np.clip((locus_end - 1 - window_start) // bin_bp, 0, WINDOW_BINS - 1)
    return (index >= first) & (index <= last)
