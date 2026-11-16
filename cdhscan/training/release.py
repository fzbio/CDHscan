"""The model folder: panel reference, catalogue, bins and weights.

Panel reference of one chromosome (format in panel.py): per scored locus, the median and MAD over
the clean panel (panel people minus the locus carriers) of the balanced and raw log2(O/E) window
maps (upper triangle), the log relative coverage and the 4 WGS values of the window's bins.

Model folder layout (what score.score_person reads):
  models/hic_wgs_seed<seed>.pt   catalogue.tsv   bins.tsv   panel_reference/<chrom>.npz
  training_summary.tsv           MD5SUMS
The catalogue keeps the number of carriers of each locus, not their names.
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import BIN_BP, SEEDS, WGS_VALUES, WINDOW_BINS
from ..features import median_and_mad, window_values
from ..panel import TRIANGLE_ARRAYS, UPPER

CATALOGUE_COLUMNS = ["locus_id", "chrom", "start", "end", "n_carriers", "window_start", "window_end",
                     "aligned_fraction", "region_start", "region_end", "grch38_start", "grch38_end",
                     "region_aligned_fraction"]


def panel_reference(loci, panel, value_bands, wgs, first_row, out_file):
    """loci: the scored loci of one chromosome (with carriers); value_bands, wgs: person -> values."""
    n_loci = len(loci)
    reference = {name: np.empty((n_loci, len(UPPER[0])), dtype=np.float32) for name in TRIANGLE_ARRAYS}
    for name in ["coverage_median", "coverage_mad"]:
        reference[name] = np.empty((n_loci, WINDOW_BINS), dtype=np.float32)
    for name in ["wgs_median", "wgs_mad"]:
        reference[name] = np.empty((n_loci, WINDOW_BINS, len(WGS_VALUES)), dtype=np.float32)
    n_clean = np.empty(n_loci, dtype=np.int32)

    for i, locus in enumerate(loci.itertuples()):
        first_bin = locus.window_start // BIN_BP
        carriers = set(locus.carriers.split(","))
        clean_people = [person for person in panel if person not in carriers]
        n_clean[i] = len(clean_people)
        balanced, raw, coverage = (np.stack(part) for part in
                                   zip(*(window_values(value_bands[person], first_bin) for person in clean_people)))
        for name, values in [("balanced", balanced), ("raw", raw)]:
            median, mad = median_and_mad(values)
            reference[f"{name}_median"][i], reference[f"{name}_mad"][i] = median[UPPER], mad[UPPER]
        reference["coverage_median"][i], reference["coverage_mad"][i] = median_and_mad(coverage)
        rows = first_row + first_bin + np.arange(WINDOW_BINS)
        reference["wgs_median"][i], reference["wgs_mad"][i] = median_and_mad(np.stack([wgs[p][rows] for p in clean_people]))
    np.savez_compressed(out_file, locus_id=loci["locus_id"].to_numpy().astype(str), n_clean=n_clean, **reference)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_model(loci_file, bins_file, seed_dirs, panel_reference_dir, chroms, out_dir):
    """Copy everything score.score_person needs into out_dir, with a training summary and md5s."""
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "panel_reference").mkdir(exist_ok=True)
    summary = []
    for seed, seed_dir in zip(SEEDS, seed_dirs):
        shutil.copyfile(Path(seed_dir) / "best_model.pt", out_dir / "models" / f"hic_wgs_seed{seed}.pt")
        log = pd.read_csv(Path(seed_dir) / "training_log.tsv", sep="\t")
        best = log.loc[log["val_loss"].idxmin()]
        summary.append({"seed": seed, "best_epoch": int(best["epoch"]), "epochs_run": len(log),
                        "val_loss": best["val_loss"], "val_auc": best["val_auc"],
                        "val_within_locus_auc": best["val_within_locus_auc"]})
    pd.DataFrame(summary).to_csv(out_dir / "training_summary.tsv", sep="\t", index=False, float_format="%.4f")
    loci = pd.read_csv(loci_file, sep="\t")
    loci[loci["scored"]][CATALOGUE_COLUMNS].to_csv(out_dir / "catalogue.tsv", sep="\t", index=False, float_format="%.4f")
    shutil.copyfile(bins_file, out_dir / "bins.tsv")
    for chrom in chroms:
        shutil.copyfile(Path(panel_reference_dir) / f"{chrom}.npz", out_dir / "panel_reference" / f"{chrom}.npz")
    files = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "MD5SUMS")
    (out_dir / "MD5SUMS").write_text("".join(f"{md5(path)}  {path.relative_to(out_dir)}\n" for path in files))
