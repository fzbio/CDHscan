"""Read a person's CHM13 Hi-C map from a .hic file as diagonal bands.

For every requested chromosome the output holds
  band          raw read-pair count of bins i and i + d (n_bins x 100); NaN past the chromosome end;
                a block without any contact record counts as zero contacts
  scale_factor  SCALE normalisation factor of every bin; NaN where SCALE has no factor
The balanced map is raw / (factor of bin i x factor of bin j), computed later from these two.
The .hic must hold 10 kb raw counts and SCALE factors (as the HPRC chm13.hic files do).
"""

from pathlib import Path

import hicstraw
import numpy as np

from .constants import BIN_BP, WINDOW_BINS

CHUNK_BINS = 500   # rows read per request


def find_chromosome(hic, chrom):
    """The .hic chromosome named chrom or chrom without the "chr" prefix (HPRC maps use 1, 2, ...)."""
    names = {chrom, chrom.removeprefix("chr")}
    matches = [c for c in hic.getChromosomes() if c.name in names]
    assert len(matches) == 1, f"{chrom}: not found in the .hic file"
    return matches[0]


def read_scale_factors(hic, chromosome, n_bins):
    zoom = hic.getMatrixZoomData(chromosome.name, chromosome.name, "observed", "SCALE", "BP", BIN_BP)
    factors = np.array(zoom.getNormVector(chromosome.index), dtype=float)
    assert len(factors) >= n_bins, f"{chromosome.name}: {len(factors)} SCALE factors for {n_bins} bins"
    factors = factors[:n_bins]
    factors[~np.isfinite(factors) | (factors == 0)] = np.nan
    return factors.astype(np.float32)


def read_raw_band(zoom, n_bins):
    band = np.full((n_bins, WINDOW_BINS), np.nan, dtype=np.float32)
    for first_row in range(0, n_bins, CHUNK_BINS):
        last_row = min(first_row + CHUNK_BINS, n_bins) - 1
        last_col = min(last_row + WINDOW_BINS - 1, n_bins - 1)
        region = (first_row * BIN_BP, last_row * BIN_BP, first_row * BIN_BP, last_col * BIN_BP)
        block = zoom.getRecordsAsMatrix(*region)
        n_rows, n_cols = last_row - first_row + 1, last_col - first_row + 1
        if block.shape == (1, 1) and len(zoom.getRecords(*region)) == 0:
            # hic-straw returns a 1 x 1 matrix for a block without any contact record (e.g. a centromere).
            block = np.zeros((n_rows, n_cols), dtype=np.float32)
        assert block.shape == (n_rows, n_cols), f"rows {first_row}-{last_row}: got {block.shape}"
        rows = np.arange(n_rows)
        for distance in range(WINDOW_BINS):
            inside = rows + distance < n_cols
            band[first_row + rows[inside], distance] = block[rows[inside], rows[inside] + distance]
    assert np.nansum(band) > 0, "only zero contacts on the whole chromosome"
    return band


def extract_bands(hic_path, chroms, out_dir):
    """Write <out_dir>/<chrom>.npz (band, scale_factor) for every chromosome of chroms."""
    hic = hicstraw.HiCFile(str(hic_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for chrom in chroms:
        chromosome = find_chromosome(hic, chrom)
        n_bins = -(-chromosome.length // BIN_BP)
        scale_factor = read_scale_factors(hic, chromosome, n_bins)
        zoom = hic.getMatrixZoomData(chromosome.name, chromosome.name, "observed", "NONE", "BP", BIN_BP)
        band = read_raw_band(zoom, n_bins)
        np.savez_compressed(out_dir / f"{chrom}.npz", band=band, scale_factor=scale_factor)
        print(f"{chrom}: {n_bins} bins, {int(np.nansum(band)):,} read pairs within 1 Mb, "
              f"{np.isfinite(scale_factor).sum()} bins with SCALE factors", flush=True)


def load_bands(band_dir, chrom):
    stored = np.load(Path(band_dir) / f"{chrom}.npz")
    return stored["band"], stored["scale_factor"]
