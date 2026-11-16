"""Keep the Hi-C contacts inside the locus windows from a pairtools parse stream.

Reads a .pairs stream (pairtools parse) on stdin and writes to stdout the header and the pairs
with both ends unique (VALID_PAIR_TYPES) on one chromosome inside one Hi-C window (overlapping
windows are merged). The windows are given in the coordinates of the reference the reads were
mapped to.

With --stage2-ids (CHM13 mapping) it also writes the read IDs that go on to the individualized
references: every pair except those with an end mapped uniquely outside all windows. Such an end
maps to the same place on an individualized reference, whose sequence outside the swapped regions
is CHM13, so the pair can never become a contact inside a window there.
"""

import bisect
import sys

from ..constants import UNIQUE_SIDE_TYPES, VALID_PAIR_TYPES


def read_windows(bed_file):
    """Merged windows per chromosome, as sorted start and end lists (0-based, end exclusive)."""
    intervals = {}
    with open(bed_file) as handle:
        for line in handle:
            chrom, start, end = line.split("\t")[:3]
            intervals.setdefault(chrom, []).append((int(start), int(end)))
    windows = {}
    for chrom, chrom_intervals in intervals.items():
        merged = []
        for start, end in sorted(chrom_intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        windows[chrom] = ([start for start, _ in merged], [end for _, end in merged])
    return windows


def window_of(windows, chrom, position):
    """Index of the merged window holding a 1-based pairs position, or -1."""
    if chrom not in windows:
        return -1
    starts, ends = windows[chrom]
    index = bisect.bisect_right(starts, position - 1) - 1
    return index if index >= 0 and position - 1 < ends[index] else -1


def filter_pairs(windows_file, stage2_ids_file=None):
    """stdin pairs stream -> stdout window contacts; optionally the read IDs for the second mapping."""
    windows = read_windows(windows_file)
    ids = open(stage2_ids_file, "w") if stage2_ids_file else None
    n_pairs = n_window = n_stage2 = 0
    for line in sys.stdin:
        if line.startswith("#"):
            if line.startswith("#columns:"):
                columns = line.split()[1:]
                read_id, chrom1, pos1, chrom2, pos2, pair_type = (
                    columns.index(name) for name in ["readID", "chrom1", "pos1", "chrom2", "pos2", "pair_type"])
            sys.stdout.write(line)
            continue
        fields = line.rstrip("\n").split("\t")
        n_pairs += 1
        window1 = window_of(windows, fields[chrom1], int(fields[pos1]))
        window2 = window_of(windows, fields[chrom2], int(fields[pos2]))
        types = fields[pair_type]
        if types in VALID_PAIR_TYPES and window1 >= 0 and fields[chrom1] == fields[chrom2] and window1 == window2:
            sys.stdout.write(line)
            n_window += 1
        if ids is not None:
            unique_outside = ((types[0] in UNIQUE_SIDE_TYPES and window1 < 0)
                              or (types[1] in UNIQUE_SIDE_TYPES and window2 < 0))
            if not unique_outside:
                ids.write(fields[read_id] + "\n")
                n_stage2 += 1
    if ids is not None:
        ids.close()
    print(f"pairs {n_pairs}, window contacts {n_window}"
          + (f", read pairs for the individualized references {n_stage2}" if ids is not None else ""), file=sys.stderr)
