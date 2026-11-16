"""The locus catalogue and the training windows.

Catalogue: the loci come from bedtools (workflow rule "catalogue"): the panel people's CHM13 CDH
regions of at most 1 Mb merged into loci of at most 1 Mb; carriers = panel people with a region of
any length overlapping the locus. Per locus this module adds
  window_start/end   the 1 Mb window centred on the locus midpoint (the same for every person)
  aligned_fraction   share of the locus aligned to GRCh38 (lift.lift_interval)
  scored             aligned_fraction >= MIN_ALIGNED_FRACTION and the window lies inside the
                     chromosome: the loci the model scores
  region_*, grch38_* the replacement region (locus +-REPLACEMENT_FLANK_BP) on CHM13 and lifted to
                     GRCh38, where cosigt picks proxy haplotypes

Training windows, only at scored loci with at least MIN_CARRIERS carriers:
  positive   a CDH region of at most 1 Mb of the person overlaps the locus
  negative   no CDH region of the person overlaps the locus
  dropped    otherwise, or when more than MAX_BLANK_FRACTION of the window's bins have no SCALE
             factor in the person's map
Train and validation windows are matched within each locus and role: as many people of each
class as the smaller class, chosen at random. The Hi-C inputs are stored per chromosome in
PARTS_PER_CHROM parts of whole loci (input_file, input_row).
"""

import numpy as np
import pandas as pd

from ..constants import (BIN_BP, LABELLED_CHROMS, MAX_BLANK_FRACTION, MIN_ALIGNED_FRACTION, MIN_CARRIERS,
                         REPLACEMENT_FLANK_BP, WINDOW_BINS)
from ..features import blank_bins_per_window
from .lift import lift_interval

MAX_REGION_BP = 1_000_000
PARTS_PER_CHROM = 4
LOCUS_COLUMNS = ["chrom", "start", "end", "locus_id", "carriers"]
WINDOW_COLUMNS = ["window_id", "sample", "role", "chrom", "start", "end", "label", "locus_id", "locus_start",
                  "locus_end", "n_carriers", "n_blank_bins", "input_file", "input_row"]


def catalogue(loci_bed, chains, n_bins):
    """The bedtools loci with carriers counted, window, scored flag and lifted replacement region."""
    loci = pd.read_csv(loci_bed, sep="\t", header=None, names=LOCUS_COLUMNS)
    loci = loci[loci["chrom"].isin(LABELLED_CHROMS)].reset_index(drop=True)
    loci["n_carriers"] = loci["carriers"].map(lambda names: len(names.split(",")))
    first_bins = (loci["start"] + loci["end"]) // 2 // BIN_BP - WINDOW_BINS // 2
    loci["window_start"], loci["window_end"] = first_bins * BIN_BP, (first_bins + WINDOW_BINS) * BIN_BP
    loci["region_start"] = (loci["start"] - REPLACEMENT_FLANK_BP).clip(lower=0)
    loci["region_end"] = loci["end"] + REPLACEMENT_FLANK_BP
    loci["aligned_fraction"] = [lift_interval(chains[row.chrom], row.start, row.end)[2] for row in loci.itertuples()]
    region_lift = [lift_interval(chains[row.chrom], row.region_start, row.region_end) for row in loci.itertuples()]
    loci["grch38_start"], loci["grch38_end"], loci["region_aligned_fraction"] = zip(*region_lift)
    inside = (first_bins >= 0) & (first_bins + WINDOW_BINS <= loci["chrom"].map(n_bins))
    loci["scored"] = inside & (loci["aligned_fraction"] >= MIN_ALIGNED_FRACTION)
    return loci


def overlapping(loci_starts, loci_ends, regions):
    """Boolean per locus: does any of the regions overlap it?"""
    if regions.empty:
        return np.zeros(len(loci_starts), dtype=bool)
    starts, ends = regions["start"].to_numpy(), regions["end"].to_numpy()
    return ((loci_starts[:, None] < ends) & (loci_ends[:, None] > starts)).any(axis=1)


def candidate_windows(loci, regions, roles, valid_bins):
    """One labelled candidate window per locus and person; unusable ones left out."""
    parts = []
    for chrom, chrom_loci in loci.groupby("chrom", sort=False):
        starts, ends = chrom_loci["start"].to_numpy(), chrom_loci["end"].to_numpy()
        first_bins = (chrom_loci["window_start"] // BIN_BP).to_numpy()
        chrom_regions = regions[regions["chrom"] == chrom]
        for person, role in zip(roles["sample"], roles["role"]):
            person_regions = chrom_regions[chrom_regions["sample"] == person]
            short_regions = person_regions[person_regions["end"] - person_regions["start"] <= MAX_REGION_BP]
            is_positive = overlapping(starts, ends, short_regions)
            is_negative = ~overlapping(starts, ends, person_regions)
            n_blank = blank_bins_per_window(valid_bins[person][chrom], first_bins)
            keep = (is_positive | is_negative) & (n_blank <= MAX_BLANK_FRACTION * WINDOW_BINS)
            parts.append(pd.DataFrame({
                "sample": person, "role": role, "chrom": chrom, "start": first_bins[keep] * BIN_BP,
                "end": (first_bins[keep] + WINDOW_BINS) * BIN_BP, "label": is_positive[keep].astype(int),
                "locus_id": chrom_loci["locus_id"].to_numpy()[keep], "locus_start": starts[keep],
                "locus_end": ends[keep], "n_carriers": chrom_loci["n_carriers"].to_numpy()[keep],
                "n_blank_bins": n_blank[keep]}))
    return pd.concat(parts, ignore_index=True)


def match_within_locus(windows, rng):
    """Per locus, as many people of each class as the smaller class; loci with one class dropped."""
    kept = []
    for _, group in windows.groupby("locus_id", sort=False):
        positives, negatives = group[group["label"] == 1], group[group["label"] == 0]
        n_per_class = min(len(positives), len(negatives))
        if n_per_class == 0:
            continue
        for people in (positives, negatives):
            chosen = rng.choice(len(people), n_per_class, replace=False)
            kept.append(people.iloc[np.sort(chosen)])
    return pd.concat(kept)


def assign_input_parts(windows):
    """Per chromosome, cut the loci in order into PARTS_PER_CHROM parts of similar window counts."""
    windows["input_file"], windows["input_row"] = "", 0
    for chrom, part_windows in windows.groupby("chrom", sort=False):
        locus_order = part_windows["locus_id"].drop_duplicates()
        locus_sizes = part_windows.groupby("locus_id").size().loc[locus_order]
        part_of_locus = (np.cumsum(locus_sizes.to_numpy()) - 1) * PARTS_PER_CHROM // len(part_windows)
        part = part_windows["locus_id"].map(dict(zip(locus_order, part_of_locus)))
        windows.loc[part_windows.index, "input_file"] = [f"fit_{chrom}_part{p + 1}.npy" for p in part]
        windows.loc[part_windows.index, "input_row"] = part_windows.groupby(part.to_numpy()).cumcount().to_numpy()
    return windows


def training_windows(loci, regions, roles, valid_bins, seed):
    """Matched train and validation windows at the scored loci with at least MIN_CARRIERS carriers."""
    fitting_loci = loci[loci["scored"] & (loci["n_carriers"] >= MIN_CARRIERS)]
    candidates = candidate_windows(fitting_loci, regions, roles, valid_bins)
    rng = np.random.default_rng(seed)
    windows = pd.concat([match_within_locus(candidates[candidates["role"] == role], rng)
                         for role in ["train", "validation"]])
    windows["chrom_order"] = windows["chrom"].map(LABELLED_CHROMS.index)
    windows = windows.sort_values(["chrom_order", "locus_start", "locus_id", "role", "sample"])
    windows = assign_input_parts(windows.drop(columns="chrom_order").reset_index(drop=True))
    windows.insert(0, "window_id", [f"W{i:07d}" for i in range(len(windows))])
    return windows[WINDOW_COLUMNS]
