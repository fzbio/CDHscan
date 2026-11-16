"""Read counts of a person's GRCh38 WGS alignment (CRAM or BAM) in every lifted 10 kb CHM13 bin.

The model works on CHM13 bins; the WGS reads are on GRCh38. Every CHM13 bin carries its GRCh38
span (bins.tsv, from the UCSC hs1ToHg38 chain); a bin with less than half of its bases aligned is
not lifted and gets no counts.

Reads count in a bin by their start position in the bin's GRCh38 span. Unmapped, secondary,
supplementary, duplicate and QC-fail reads never count. Per bin (COUNT_COLUMNS):
  depth_reads    reads with mapping quality >= MIN_MAPQ (sequencing depth; drops at a deletion)
  all_reads      reads of any mapping quality
  soft_clipped   reads with >= MIN_CLIP_BP soft-clipped bases at either end   (any mapping quality)
  mate_unmapped  paired reads whose mate did not map                        (any mapping quality)
  discordant     paired reads whose mate mapped, but not as a proper pair   (any mapping quality)
  *_q20          the last three among the depth_reads
The alignment is read in chunks of touching GRCh38 bins; each bin's counts are differences of
running totals at its two ends.
"""

import numpy as np
import pandas as pd
import pysam

from .constants import BIN_BP, COUNT_COLUMNS, MIN_CLIP_BP, MIN_MAPQ

SOFT_CLIP = 4      # CIGAR operation code
CHUNK_BINS = 200   # about 2 Mb of reads held in memory at a time


def parse_region(region):
    """'chr20:30000000-33000000' -> ('chr20', 30000000, 33000000)."""
    chrom, span = region.split(":")
    start, end = (int(value.replace(",", "")) for value in span.split("-"))
    return chrom, start, end


def load_bins(path, chroms, region=None):
    """bins.tsv rows of the given chromosomes. With a CHM13 region, bins outside it count as not
    lifted, so they get no reads and stay out of the depth normalisation."""
    bins = pd.read_csv(path, sep="\t")
    bins = bins[bins["chrom"].isin(chroms)].reset_index(drop=True)
    if region is not None:
        chrom, start, end = parse_region(region)
        inside = (bins["chrom"] == chrom) & (bins["bin"] * BIN_BP >= start) & ((bins["bin"] + 1) * BIN_BP <= end)
        bins["lifted"] = bins["lifted"] & inside
    return bins


def clipped(read):
    cigar = read.cigartuples
    return (cigar[0][0] == SOFT_CLIP and cigar[0][1] >= MIN_CLIP_BP) or \
           (cigar[-1][0] == SOFT_CLIP and cigar[-1][1] >= MIN_CLIP_BP)


def read_flags(alignment, chrom, start, end):
    """Sorted start positions of the counted reads in [start, end), and their flags (COUNT_COLUMNS order)."""
    starts, flags = [], []
    for read in alignment.fetch(chrom, start, end):
        if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail:
            continue
        mate_unmapped = read.is_paired and read.mate_is_unmapped
        discordant = read.is_paired and not read.mate_is_unmapped and not read.is_proper_pair
        is_clipped = clipped(read)
        good_mapping = read.mapping_quality >= MIN_MAPQ
        starts.append(read.reference_start)
        flags.append((good_mapping, True, is_clipped, mate_unmapped, discordant,
                      good_mapping and is_clipped, good_mapping and mate_unmapped, good_mapping and discordant))
    starts = np.array(starts, dtype=np.int64)
    assert (np.diff(starts) >= 0).all(), f"{chrom}:{start}-{end}: reads are not sorted by position"
    return starts, np.array(flags, dtype=bool).reshape(-1, len(COUNT_COLUMNS))


def read_chunks(bins):
    """Chunk number of each bin (one chromosome, sorted by GRCh38 start): stretches of bins whose
    GRCh38 spans overlap or touch, cut into pieces of at most CHUNK_BINS bins."""
    stretch_end = bins["grch38_end"].cummax().to_numpy()
    new_stretch = bins["grch38_start"].to_numpy()[1:] > stretch_end[:-1]
    stretch = np.concatenate([[0], np.cumsum(new_stretch)])
    position_in_stretch = pd.Series(stretch).groupby(stretch).cumcount().to_numpy()
    new_chunk = np.concatenate([[True], (stretch[1:] != stretch[:-1]) | (position_in_stretch[1:] % CHUNK_BINS == 0)])
    return np.cumsum(new_chunk)


def count_reads(alignment_path, reference, bins):
    """Counts (n_bins x 8, int32) in bins order; bins that are not lifted stay 0."""
    counts = np.zeros((len(bins), len(COUNT_COLUMNS)), dtype=np.int32)
    with pysam.AlignmentFile(str(alignment_path), reference_filename=reference) as alignment:
        for chrom, chrom_bins in bins[bins["lifted"]].groupby("chrom", sort=False):
            chrom_bins = chrom_bins.sort_values("grch38_start")
            for _, chunk in chrom_bins.groupby(read_chunks(chrom_bins)):
                starts, flags = read_flags(alignment, chrom, chunk["grch38_start"].min(), chunk["grch38_end"].max())
                running = np.vstack([np.zeros(len(COUNT_COLUMNS), dtype=np.int64), np.cumsum(flags, axis=0)])
                first = np.searchsorted(starts, chunk["grch38_start"].to_numpy(), side="left")
                after_last = np.searchsorted(starts, chunk["grch38_end"].to_numpy(), side="left")
                counts[chunk.index] = running[after_last] - running[first]
            print(f"{chrom}: {len(chrom_bins)} bins counted", flush=True)
    return counts
