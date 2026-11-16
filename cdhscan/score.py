"""Score every catalogue locus of one person with the released Hi-C + WGS model.

Inputs of the person: the Hi-C bands (hic.extract_bands) and the WGS read counts (reads.count_reads)
on the bins of the released bins.tsv. Per locus, the model window is the 1 Mb window centred on
the locus. A window is scored only when at most MAX_BLANK_FRACTION of its bins lack a SCALE
factor in the person's map. Per scored window:
  Hi-C input   12 x 100 x 100: own map values, own minus the clean panel median, clean panel MAD
               (rounded to float16, the precision of the training inputs); plus the locus mask
  WGS input    12 x 100: the same three parts for the 4 WGS values; plus the locus mask
NaN values become 0. Each input channel is standardised with the training mean and sd stored in
each checkpoint. The probability of a locus is the mean over the seeds; it is a CDH call when it
exceeds CDH_PROBABILITY.

Output columns: locus_id, chrom, start, end, n_carriers (among the 171 panel people), window_start,
window_end, n_blank_bins, status (scored / too_many_blank_bins), probability_seed<seed>,
probability, cdh.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import (BIN_BP, CDH_PROBABILITY, MAP_CHANNELS, MAX_BLANK_FRACTION, SEEDS, WGS_CHANNELS,
                        WINDOW_BINS)
from .features import (as_map_channels, blank_bins_per_window, hic_value_bands, locus_mask, wgs_values,
                       window_values)
from .hic import load_bands
from .model import JointCNN
from .panel import load_panel
from .reads import load_bins, parse_region

BATCH_SIZE = 128


class ReleasedModel:
    """One seed of the released model with its input standardisation."""

    def __init__(self, path):
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["map_channels"] == MAP_CHANNELS and checkpoint["wgs_channels"] == WGS_CHANNELS, \
            f"{path}: channels differ from this version of CDHscan"
        self.network = JointCNN(n_map_channels=len(MAP_CHANNELS) + 1, n_track_channels=len(WGS_CHANNELS) + 1)
        self.network.load_state_dict(checkpoint["model"])
        self.network.eval()
        self.statistics = {name: value.numpy() for name, value in checkpoint["statistics"].items()}

    def probability(self, maps, tracks, in_locus):
        """maps: n x 12 x 100 x 100, tracks: n x 12 x 100, in_locus: n x 100 -> n probabilities."""
        s = self.statistics
        maps = (maps.astype(np.float32) - s["map_mean"][:, None, None]) / s["map_sd"][:, None, None]
        tracks = (tracks - s["track_mean"][:, None]) / s["track_sd"][:, None]
        map_mask = (in_locus[:, :, None] | in_locus[:, None, :]).astype(np.float32)
        contact_map = torch.from_numpy(np.concatenate([maps, map_mask[:, None]], axis=1).astype(np.float32))
        track_input = torch.from_numpy(np.concatenate([tracks, in_locus[:, None].astype(np.float32)], axis=1))
        with torch.no_grad():
            return torch.sigmoid(self.network(contact_map, track_input)).numpy()


def catalogue_loci(model_dir, chroms, region):
    """Catalogue loci on the given chromosomes; with a region, only loci whose window lies inside it."""
    loci = pd.read_csv(Path(model_dir) / "catalogue.tsv", sep="\t")
    loci = loci[loci["chrom"].isin(chroms)]
    if region is not None:
        chrom, start, end = parse_region(region)
        loci = loci[(loci["chrom"] == chrom) & (loci["window_start"] >= start) & (loci["window_end"] <= end)]
    return loci.reset_index(drop=True)


def chromosome_inputs(loci, value_bands, wgs, first_row, panel):
    """Hi-C inputs (float16), WGS inputs and locus masks of the loci of one chromosome."""
    maps = np.empty((len(loci), len(MAP_CHANNELS), WINDOW_BINS, WINDOW_BINS), dtype=np.float16)
    tracks = np.empty((len(loci), len(WGS_CHANNELS), WINDOW_BINS), dtype=np.float32)
    in_locus = np.empty((len(loci), WINDOW_BINS), dtype=bool)
    for i, locus in enumerate(loci.itertuples()):
        first_bin = locus.window_start // BIN_BP
        own = as_map_channels(*window_values(value_bands, first_bin))
        median, mad = panel.map_statistics(locus.locus_id)
        maps[i] = np.nan_to_num(np.concatenate([own, own - median, mad]), nan=0.0).astype(np.float16)
        own_wgs = wgs[first_row + first_bin + np.arange(WINDOW_BINS)]
        wgs_median, wgs_mad = panel.wgs_statistics(locus.locus_id)
        tracks[i] = np.nan_to_num(np.concatenate([own_wgs, own_wgs - wgs_median, wgs_mad], axis=1), nan=0.0).T
        in_locus[i] = locus_mask(locus.start, locus.end, locus.window_start, BIN_BP)
    return maps, tracks, in_locus


def predict_in_batches(model, maps, tracks, in_locus):
    probabilities = [np.empty(0, dtype=np.float32)]
    for i in range(0, len(maps), BATCH_SIZE):
        part = slice(i, i + BATCH_SIZE)
        probabilities.append(model.probability(maps[part], tracks[part], in_locus[part]))
    return np.concatenate(probabilities)


def score_person(model_dir, band_dir, counts_file, chroms, region=None):
    model_dir = Path(model_dir)
    models = {seed: ReleasedModel(model_dir / "models" / f"hic_wgs_seed{seed}.pt") for seed in SEEDS}
    bins = load_bins(model_dir / "bins.tsv", chroms, region)
    counts = np.load(counts_file)
    assert counts.shape[0] == len(bins), "read counts were made on other bins (chromosomes or region differ)"
    wgs = wgs_values(counts, bins)
    first_rows = bins.reset_index().groupby("chrom")["index"].min().to_dict()
    loci = catalogue_loci(model_dir, chroms, region)

    tables = []
    for chrom, chrom_loci in loci.groupby("chrom", sort=False):
        raw_band, scale_factor = load_bands(band_dir, chrom)
        first_bins = (chrom_loci["window_start"] // BIN_BP).to_numpy()
        chrom_loci = chrom_loci.assign(n_blank_bins=blank_bins_per_window(np.isfinite(scale_factor), first_bins))
        chrom_loci["status"] = np.where(chrom_loci["n_blank_bins"] <= MAX_BLANK_FRACTION * WINDOW_BINS,
                                        "scored", "too_many_blank_bins")
        scored = chrom_loci[chrom_loci["status"] == "scored"]
        maps, tracks, in_locus = chromosome_inputs(scored, hic_value_bands(raw_band, scale_factor), wgs,
                                                   first_rows[chrom], load_panel(model_dir / "panel_reference", chrom))
        for seed, model in models.items():
            column = np.full(len(chrom_loci), np.nan)
            column[(chrom_loci["status"] == "scored").to_numpy()] = predict_in_batches(model, maps, tracks, in_locus)
            chrom_loci[f"probability_seed{seed}"] = column
        tables.append(chrom_loci)
        print(f"{chrom}: {len(scored)} of {len(chrom_loci)} loci scored", flush=True)

    table = pd.concat(tables, ignore_index=True)
    table["probability"] = table[[f"probability_seed{seed}" for seed in SEEDS]].mean(axis=1)
    table["cdh"] = table["probability"] > CDH_PROBABILITY
    columns = ["locus_id", "chrom", "start", "end", "n_carriers", "window_start", "window_end", "n_blank_bins",
               "status"] + [f"probability_seed{seed}" for seed in SEEDS] + ["probability", "cdh"]
    return table[columns]
