"""CHM13 to GRCh38 lifting with the UCSC hs1ToHg38 chain file.

Only plus-strand chains between the same chromosome are used. An interval is lifted through the
chain with the most aligned bases in it; its GRCh38 span runs from the first to the last aligned
base, and its aligned fraction is the share of its bases aligned in that chain. A 10 kb bin with an
aligned fraction below MIN_ALIGNED_FRACTION is not lifted (the WGS reads cannot measure it).
"""

import gzip

import numpy as np
import pandas as pd

from ..constants import BIN_BP, MIN_ALIGNED_FRACTION


def read_chains(chain_file, chroms):
    """Per chromosome, one (n_blocks x 3) array per chain: CHM13 block starts, CHM13 block ends,
    GRCh38 block starts."""
    chains = {chrom: [] for chrom in chroms}
    current = None
    with gzip.open(chain_file, "rt") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "chain":
                t_name, t_strand, t_start = fields[2], fields[4], int(fields[5])
                q_name, q_strand, q_start = fields[7], fields[9], int(fields[10])
                keep = t_name in chains and q_name == t_name and t_strand == "+" and q_strand == "+"
                current = {"t": t_start, "q": q_start, "blocks": []} if keep else None
                if keep:
                    chains[t_name].append(current)
                continue
            if current is None:
                continue
            size = int(fields[0])
            current["blocks"].append((current["t"], current["t"] + size, current["q"]))
            if len(fields) == 3:
                current["t"] += size + int(fields[1])
                current["q"] += size + int(fields[2])
    return {chrom: [np.array(chain["blocks"]) for chain in chrom_chains] for chrom, chrom_chains in chains.items()}


def lift_interval(blocks_of_chains, start, end):
    """(GRCh38 start, GRCh38 end, aligned fraction) of CHM13 [start, end); (-1, -1, 0) if nothing aligns."""
    best = (0, -1, -1)
    for blocks in blocks_of_chains:
        t_starts, t_ends, q_starts = blocks[:, 0], blocks[:, 1], blocks[:, 2]
        overlap_starts, overlap_ends = np.maximum(t_starts, start), np.minimum(t_ends, end)
        overlapping = overlap_ends > overlap_starts
        aligned = int((overlap_ends - overlap_starts)[overlapping].sum())
        if aligned > best[0]:
            q_low = int((q_starts + overlap_starts - t_starts)[overlapping].min())
            q_high = int((q_starts + overlap_ends - t_starts)[overlapping].max())
            best = (aligned, q_low, q_high)
    return best[1], best[2], best[0] / (end - start)


def chain_pieces(blocks, chain_number, n_bins):
    """The parts of a chain's blocks inside each 10 kb bin: bin, chain, aligned bases, GRCh38 span."""
    t_starts, t_ends, q_starts = blocks[:, 0], blocks[:, 1], blocks[:, 2]
    first_bins, last_bins = t_starts // BIN_BP, (t_ends - 1) // BIN_BP
    n_pieces = last_bins - first_bins + 1
    # One row per (block, bin it touches): a long block is cut at the bin borders.
    block = np.repeat(np.arange(len(blocks)), n_pieces)
    bin_index = first_bins[block] + np.arange(n_pieces.sum()) - np.repeat(np.cumsum(n_pieces) - n_pieces, n_pieces)
    inside = bin_index < n_bins
    block, bin_index = block[inside], bin_index[inside]
    piece_starts = np.maximum(t_starts[block], bin_index * BIN_BP)
    piece_ends = np.minimum(t_ends[block], (bin_index + 1) * BIN_BP)
    to_grch38 = q_starts[block] - t_starts[block]
    return pd.DataFrame({"bin": bin_index, "chain": chain_number, "aligned": piece_ends - piece_starts,
                         "grch38_start": piece_starts + to_grch38, "grch38_end": piece_ends + to_grch38})


def lift_bins(blocks_of_chains, n_bins):
    """GRCh38 start, end and aligned fraction of every bin of a chromosome in its best chain."""
    pieces = pd.concat([chain_pieces(blocks, number, n_bins) for number, blocks in enumerate(blocks_of_chains)])
    per_chain = pieces.groupby(["bin", "chain"]).agg(aligned=("aligned", "sum"), grch38_start=("grch38_start", "min"),
                                                     grch38_end=("grch38_end", "max")).reset_index()
    # Ties go to the earlier chain in the file.
    best = per_chain.sort_values(["aligned", "chain"], ascending=[False, True]).drop_duplicates("bin")
    lifted = pd.DataFrame({"bin": np.arange(n_bins)}).merge(best.drop(columns="chain"), on="bin", how="left")
    lifted["aligned_fraction"] = lifted["aligned"].fillna(0) / BIN_BP
    lifted["lifted"] = lifted["aligned_fraction"] >= MIN_ALIGNED_FRACTION
    lifted.loc[~lifted["lifted"], ["grch38_start", "grch38_end"]] = -1
    lifted[["grch38_start", "grch38_end"]] = lifted[["grch38_start", "grch38_end"]].astype(int)
    return lifted[["bin", "grch38_start", "grch38_end", "aligned_fraction", "lifted"]]


def bin_table(chain_file, n_bins):
    """bins.tsv of the chromosomes in n_bins (chrom -> number of 10 kb bins), in that order."""
    chains = read_chains(chain_file, list(n_bins))
    bins = pd.concat([lift_bins(chains[chrom], count).assign(chrom=chrom) for chrom, count in n_bins.items()],
                     ignore_index=True)
    return bins[["chrom", "bin", "grch38_start", "grch38_end", "aligned_fraction", "lifted"]]
