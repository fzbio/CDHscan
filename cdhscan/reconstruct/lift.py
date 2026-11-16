"""Position conversion between an individualized reference and CHM13.

An individualized reference is CHM13 with some intervals swapped for allele sequences. Outside the
swapped intervals a position only shifts by the summed length change of the swaps to its left.
Inside a swapped interval the allele's alignment to CHM13 gives the conversion: the alignment
is stored as matching points (allele offset, CHM13 position) at the ends of every gap-free
block, and positions between points are interpolated. Allele bases without a CHM13 counterpart
(an insertion) therefore land on the CHM13 position where the insertion sits, and CHM13 bases
missing from the allele (a deletion) land on the allele base after the deletion. Beyond the
first and last points the conversion continues with slope 1.
"""

import numpy as np
import pandas as pd


def interpolate(x, xp, fp):
    """np.interp with slope-1 extension beyond the first and last points."""
    y = np.interp(x, xp, fp)
    y = np.where(x < xp[0], fp[0] - (xp[0] - x), y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1]), y)


class ReferenceLift:
    """Converts 0-based positions between one individualized reference and CHM13, per chromosome."""

    def __init__(self, segments, points):
        self.chroms = {}
        for chrom, chrom_segments in segments.sort_values("ref_start").groupby("chrom"):
            length_change = ((chrom_segments["ref_end"] - chrom_segments["ref_start"])
                             - (chrom_segments["chm13_end"] - chrom_segments["chm13_start"])).to_numpy()
            self.chroms[chrom] = {
                "ref_start": chrom_segments["ref_start"].to_numpy(),
                "ref_end": chrom_segments["ref_end"].to_numpy(),
                "chm13_start": chrom_segments["chm13_start"].to_numpy(),
                "chm13_end": chrom_segments["chm13_end"].to_numpy(),
                "shift": np.cumsum(length_change),
                "allele_points": [points[f"allele_{locus}"] for locus in chrom_segments["locus_id"]],
                "chm13_points": [points[f"chm13_{locus}"] for locus in chrom_segments["locus_id"]],
            }

    @classmethod
    def from_files(cls, segments_file, points_file):
        return cls(pd.read_csv(segments_file, sep="\t"), np.load(points_file))

    @classmethod
    def identity(cls):
        return cls(pd.DataFrame(columns=["chrom", "locus_id", "ref_start", "ref_end", "chm13_start", "chm13_end"]), {})

    def _convert(self, chrom, positions, to_chm13):
        positions = np.asarray(positions, dtype=np.int64)
        if chrom not in self.chroms:
            return positions.copy()
        layout = self.chroms[chrom]
        starts, ends = (layout["ref_start"], layout["ref_end"]) if to_chm13 else (layout["chm13_start"], layout["chm13_end"])
        segment = np.searchsorted(starts, positions, side="right") - 1
        clipped = np.clip(segment, 0, None)
        shift = np.where(segment >= 0, layout["shift"][clipped], 0)
        converted = positions - shift if to_chm13 else positions + shift
        inside = (segment >= 0) & (positions < ends[clipped])
        for k in np.unique(segment[inside]):
            mask = inside & (segment == k)
            if to_chm13:
                offsets = positions[mask] - layout["ref_start"][k]
                converted[mask] = np.floor(interpolate(offsets, layout["allele_points"][k], layout["chm13_points"][k]))
            else:
                offsets = interpolate(positions[mask], layout["chm13_points"][k], layout["allele_points"][k])
                converted[mask] = layout["ref_start"][k] + np.floor(offsets).astype(np.int64)
        return converted

    def to_chm13(self, chrom, positions):
        return self._convert(chrom, positions, to_chm13=True)

    def to_reference(self, chrom, positions):
        return self._convert(chrom, positions, to_chm13=False)
