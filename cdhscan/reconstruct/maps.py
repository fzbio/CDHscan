"""Window contact maps of one person on CHM13 and on the two proxy references.

Inputs: per arm (MAP_ARMS: chm13, proxy1, proxy2) the person's deduplicated window contacts
(pairtools dedup keeps the non-duplicate pairs; the duplicate counts are in the dedup_stats
files), and per proxy reference its swapped segments, alignment
points and windows (reference.py). Every contact end is converted to CHM13 coordinates (lift.py):
outside the swapped regions by the length shift, inside them through the allele's alignment
(allele-only sequence lands at its insertion point). The reference's own coordinates are kept too.

Haplotype-specific contacts (Ni et al.): a contact of one proxy reference whose read ID is absent
from the other proxy reference's non-duplicate window contacts.

Per reconstructed locus:
  - maps of every arm on the 10 kb CHM13 bins of the 1 Mb model window (non-duplicate contacts)
  - haplotype-specific contacts of both proxies on CHM13 bins, and of each proxy in its own
    coordinates (10 kb bins from the window start on that reference, without conversion)
  - haplotype-specific profiles on 25 kb CHM13 bins (contacts touching a bin, each read once)
  - read pairs with a contact end in the replacement region on CHM13 and on the proxy pair

Outputs in <out-dir>:
  maps.npz       locus_ids, arms; maps (loci x arms x 100 x 100); specific_maps (loci x 100 x 100);
                 specific_maps_own (loci x 2 x 100 x 100); specific_total, specific_count (loci x 40)
  contacts.tsv   per locus and arm: contacts, contacts with an end in the region
  specific.tsv   per locus: contacts, haplotype-specific contacts, and both over the contacts
                 touching the locus
  region.tsv     per locus: read pairs touching the region on CHM13, on the proxy pair, on the proxy
                 pair only (gained by reconstruction) and on CHM13 only (lost by reconstruction)
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import BIN_BP, MAP_ARMS, PROXY_SLOTS, SPECIFIC_BIN_BP, VALID_PAIR_TYPES, WINDOW_BINS
from .lift import ReferenceLift

READ_CHUNK_LINES = 5_000_000
MAP_BIN_BP = BIN_BP
WINDOW_BP = WINDOW_BINS * BIN_BP
N_MAP_BINS = WINDOW_BINS
N_SPECIFIC_BINS = WINDOW_BP // SPECIFIC_BIN_BP


def pairs_layout(pairs_file):
    """Number of header lines and column names of a .pairs.gz file."""
    n_header = 0
    with gzip.open(pairs_file, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            n_header += 1
            if line.startswith("#columns:"):
                columns = line.split()[1:]
    return n_header, columns


def read_arm(pairs_file, lift):
    """Window contacts of one arm, by chromosome, sorted by the low CHM13 end: read hash, CHM13 ends,
    own-coordinate ends and validity (pair type UU, UR or RU)."""
    n_header, columns = pairs_layout(pairs_file)
    parts = []
    for chunk in pd.read_csv(pairs_file, sep="\t", header=None, names=columns, skiprows=n_header,
                             usecols=["readID", "chrom1", "pos1", "pos2", "pair_type"],
                             dtype={"readID": str, "chrom1": str, "pair_type": str}, chunksize=READ_CHUNK_LINES):
        parts.append(pd.DataFrame({
            "read_hash": pd.util.hash_pandas_object(chunk["readID"], index=False).to_numpy(),
            "chrom": chunk["chrom1"].to_numpy(),
            "pos1": chunk["pos1"].to_numpy() - 1,
            "pos2": chunk["pos2"].to_numpy() - 1,
            "valid": chunk["pair_type"].isin(VALID_PAIR_TYPES).to_numpy()}))
    contacts = pd.concat(parts, ignore_index=True)
    by_chrom = {}
    for chrom, rows in contacts.groupby("chrom"):
        pos1, pos2 = rows["pos1"].to_numpy(), rows["pos2"].to_numpy()
        end1, end2 = lift.to_chm13(chrom, pos1), lift.to_chm13(chrom, pos2)
        low, high = np.minimum(end1, end2), np.maximum(end1, end2)
        order = np.argsort(low, kind="stable")
        by_chrom[chrom] = {"read_hash": rows["read_hash"].to_numpy()[order], "low": low[order], "high": high[order],
                           "own_low": np.minimum(pos1, pos2)[order], "own_high": np.maximum(pos1, pos2)[order],
                           "valid": rows["valid"].to_numpy()[order]}
    return by_chrom, np.unique(contacts["read_hash"].to_numpy())


def mark_absent(arm, other_arm_reads, key):
    """Flag, for every contact of one arm, whether its read ID is absent from another arm's read IDs."""
    chroms = list(arm)
    absent = ~np.isin(np.concatenate([arm[chrom]["read_hash"] for chrom in chroms]), other_arm_reads)
    offsets = np.cumsum([0] + [len(arm[chrom]["read_hash"]) for chrom in chroms])
    for chrom, start, end in zip(chroms, offsets[:-1], offsets[1:]):
        arm[chrom][key] = absent[start:end]


def in_window(arm, chrom, start, end):
    """Contacts of one arm with both CHM13 ends in [start, end) of a chromosome."""
    if chrom not in arm:
        template = next(iter(arm.values()))
        return {key: values[:0] for key, values in template.items()}
    rows = arm[chrom]
    first, last = np.searchsorted(rows["low"], [start, end])
    keep = slice(first, last)
    inside = rows["high"][keep] < end
    return {key: values[keep][inside] for key, values in rows.items()}


def contact_map(low, high, window_start):
    """Symmetric contact count matrix on the 10 kb bins of the window."""
    bins1, bins2 = (low - window_start) // MAP_BIN_BP, (high - window_start) // MAP_BIN_BP
    counts = np.bincount(bins1 * N_MAP_BINS + bins2, minlength=N_MAP_BINS ** 2).reshape(N_MAP_BINS, N_MAP_BINS)
    return counts + counts.T - np.diag(np.diag(counts))


def touched_bins(low, high, window_start, weights):
    """Per 25 kb bin, the summed weight of the contacts touching it (a contact inside one bin counts once)."""
    bins1, bins2 = (low - window_start) // SPECIFIC_BIN_BP, (high - window_start) // SPECIFIC_BIN_BP
    return (np.bincount(bins1, weights=weights, minlength=N_SPECIFIC_BINS)
            + np.bincount(bins2, weights=weights * (bins2 != bins1), minlength=N_SPECIFIC_BINS))


def touches(low, high, start, end):
    return ((low >= start) & (low < end)) | ((high >= start) & (high < end))


def region_reads(contacts, start, end):
    """Read hashes of the valid contacts with an end in [start, end) (CHM13 coordinates)."""
    valid = contacts["valid"]
    low, high = contacts["low"][valid], contacts["high"][valid]
    return np.unique(contacts["read_hash"][valid][touches(low, high, start, end)])


def region_comparison(window, locus):
    """Read pairs with a contact end in the replacement region, on CHM13 and on the proxy pair."""
    reads = {arm: region_reads(window[arm], locus.region_start, locus.region_end) for arm in MAP_ARMS}
    proxy = np.union1d(reads["proxy1"], reads["proxy2"])
    return {"locus_id": locus.locus_id, "n_chm13": len(reads["chm13"]), "n_proxy": len(proxy),
            "n_proxy_only": len(np.setdiff1d(proxy, reads["chm13"])),
            "n_chm13_only": len(np.setdiff1d(reads["chm13"], proxy))}


def window_maps(loci_file, pairs_dir, references_dir, out_dir):
    references_dir, out_dir = Path(references_dir), Path(out_dir)
    loci = pd.read_csv(loci_file, sep="\t")
    loci = loci[loci["used"]].reset_index(drop=True)
    arms, present, own_window_start = {}, {}, {}
    for arm in MAP_ARMS:
        if arm == "chm13":
            lift = ReferenceLift.identity()
        else:
            reference = references_dir / arm
            lift = ReferenceLift.from_files(reference / "segments.tsv", reference / "points.npz")
            windows = pd.read_csv(reference / "windows.bed", sep="\t", header=None,
                                  names=["chrom", "start", "end", "locus_id"])
            own_window_start[arm] = dict(zip(windows["locus_id"], windows["start"]))
        arms[arm], present[arm] = read_arm(Path(pairs_dir) / f"{arm}.dedup.pairs.gz", lift)
        print(f"{arm}: {len(present[arm])} read pairs with window contacts", flush=True)
    proxy_a, proxy_b = PROXY_SLOTS
    mark_absent(arms[proxy_a], present[proxy_b], f"absent_in_{proxy_b}")
    mark_absent(arms[proxy_b], present[proxy_a], f"absent_in_{proxy_a}")

    maps = np.zeros((len(loci), len(MAP_ARMS), N_MAP_BINS, N_MAP_BINS), dtype=np.int32)
    specific_maps = np.zeros((len(loci), N_MAP_BINS, N_MAP_BINS), dtype=np.int32)
    specific_maps_own = np.zeros((len(loci), len(PROXY_SLOTS), N_MAP_BINS, N_MAP_BINS), dtype=np.int32)
    specific_total = np.zeros((len(loci), N_SPECIFIC_BINS), dtype=np.int32)
    specific_count = np.zeros_like(specific_total)
    contact_rows, specific_rows, region_rows = [], [], []
    for i, locus in loci.iterrows():
        window = {arm: in_window(arms[arm], locus.chrom, locus.window_start, locus.window_end) for arm in MAP_ARMS}
        for j, arm in enumerate(MAP_ARMS):
            contacts = window[arm]
            valid = contacts["valid"]
            low, high = contacts["low"][valid], contacts["high"][valid]
            maps[i, j] = contact_map(low, high, locus.window_start)
            contact_rows.append({"locus_id": locus.locus_id, "arm": arm, "n_contacts": int(valid.sum()),
                                 "n_region_contacts": int(touches(low, high, locus.region_start, locus.region_end).sum())})
        region_rows.append(region_comparison(window, locus))

        for arm, other in [(proxy_a, proxy_b), (proxy_b, proxy_a)]:
            contacts = window[arm]
            specific = contacts["valid"] & contacts[f"absent_in_{other}"]
            specific_maps[i] += contact_map(contacts["low"][specific], contacts["high"][specific], locus.window_start)
            own_start = own_window_start[arm][locus.locus_id]
            own_low = contacts["own_low"][specific] - own_start
            own_high = contacts["own_high"][specific] - own_start
            inside = (own_low >= 0) & (own_high < WINDOW_BP)
            specific_maps_own[i, PROXY_SLOTS.index(arm)] = contact_map(own_low[inside], own_high[inside], 0)

        # Profiles: every read pair once, from proxy1, or from proxy2 when proxy1 does not have it.
        a = {key: values[window[proxy_a]["valid"]] for key, values in window[proxy_a].items()}
        b = {key: values[window[proxy_b]["valid"]] for key, values in window[proxy_b].items()}
        b_only = ~np.isin(b["read_hash"], a["read_hash"])
        low = np.concatenate([a["low"], b["low"][b_only]])
        high = np.concatenate([a["high"], b["high"][b_only]])
        specific = np.concatenate([a[f"absent_in_{proxy_b}"], b[f"absent_in_{proxy_a}"][b_only]]).astype(float)
        specific_total[i] = touched_bins(low, high, locus.window_start, np.ones(len(low)))
        specific_count[i] = touched_bins(low, high, locus.window_start, specific)
        first_locus_bin = max(0, (locus.start - locus.window_start) // SPECIFIC_BIN_BP)
        last_locus_bin = min(N_SPECIFIC_BINS - 1, (locus.end - 1 - locus.window_start) // SPECIFIC_BIN_BP)
        bins1, bins2 = (low - locus.window_start) // SPECIFIC_BIN_BP, (high - locus.window_start) // SPECIFIC_BIN_BP
        touches_locus = (((bins1 >= first_locus_bin) & (bins1 <= last_locus_bin))
                         | ((bins2 >= first_locus_bin) & (bins2 <= last_locus_bin)))
        specific_rows.append({"locus_id": locus.locus_id, "n_contacts": len(low), "n_specific": int(specific.sum()),
                              "n_locus_contacts": int(touches_locus.sum()),
                              "n_locus_specific": int(specific[touches_locus].sum())})

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "maps.npz", locus_ids=loci["locus_id"].to_numpy(dtype=str), arms=np.array(MAP_ARMS),
                        maps=maps, specific_maps=specific_maps, specific_maps_own=specific_maps_own,
                        specific_total=specific_total, specific_count=specific_count)
    pd.DataFrame(contact_rows).to_csv(out_dir / "contacts.tsv", sep="\t", index=False)
    pd.DataFrame(specific_rows).to_csv(out_dir / "specific.tsv", sep="\t", index=False)
    pd.DataFrame(region_rows).to_csv(out_dir / "region.tsv", sep="\t", index=False)
    print(f"{len(loci)} loci reconstructed", flush=True)
