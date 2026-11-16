"""Build one individualized reference: CHM13 with every reconstructed region swapped for one proxy allele.

For each locus of the plan (plan.py) the allele is aligned to its CHM13 region +-ALIGNMENT_SLACK_BP
with minimap2 (asm10, forward strand: the alleles are stored in the reference orientation).
Alignments are taken longest first, as long as they do not overlap the taken ones on the allele
and keep the order of the taken ones on CHM13. The swapped CHM13 interval runs from the CHM13
position of the allele's first base to that of its last base, so the flanks join the allele where
they match it.

Outputs in <out-dir>:
  reference.fa    the individualized reference (chromosomes in CHM13 order and names; one line each)
  chrom.sizes     its chromosome lengths
  segments.tsv    per swapped locus: CHM13 interval, reference interval, allele, aligned bases
  points.npz      alignment points per locus ("allele_<locus>", "chm13_<locus>"), see lift.py
  windows.bed     the 1 Mb windows of the used loci in reference coordinates (contact filter)
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from ..constants import ALIGNMENT_SLACK_BP
from .lift import ReferenceLift, interpolate

CIGAR_OP = re.compile(r"(\d+)([=XIDM])")


def block_points(cigar, allele_start, chm13_start):
    """Matching points at both ends of every gap-free block of one alignment."""
    allele_points, chm13_points = [], []
    allele, chm13, run = allele_start, chm13_start, 0
    for length, op in CIGAR_OP.findall(cigar) + [("0", "I")]:
        length = int(length)
        if op in "=XM":
            run += length
            continue
        if run:
            allele_points += [allele, allele + run - 1]
            chm13_points += [chm13, chm13 + run - 1]
            allele, chm13, run = allele + run, chm13 + run, 0
        if op == "I":
            allele += length
        else:
            chm13 += length
    return allele_points, chm13_points


def align_allele(allele, target, target_start, tmp_dir):
    """Chained forward-strand alignments of an allele to a CHM13 target: points and matched bases."""
    (tmp_dir / "allele.fa").write_text(f">allele\n{allele}\n")
    (tmp_dir / "target.fa").write_text(f">target\n{target}\n")
    paf = subprocess.run(["minimap2", "-x", "asm10", "-c", "--eqx", "--secondary=no", "-t", "1",
                          str(tmp_dir / "target.fa"), str(tmp_dir / "allele.fa")],
                         check=True, capture_output=True, text=True).stdout
    hits = []
    for line in paf.splitlines():
        fields = line.split("\t")
        if fields[4] != "+":
            continue
        cigar = next(tag[5:] for tag in fields[12:] if tag.startswith("cg:Z:"))
        hits.append({"qs": int(fields[2]), "qe": int(fields[3]), "ts": int(fields[7]), "te": int(fields[8]),
                     "block": int(fields[10]), "cigar": cigar})
    taken = []
    for hit in sorted(hits, key=lambda h: -h["block"]):
        if all((hit["qs"] >= t["qe"] and hit["ts"] >= t["te"]) or (hit["qe"] <= t["qs"] and hit["te"] <= t["ts"])
               for t in taken):
            taken.append(hit)
    assert taken, "allele does not align to its CHM13 region"
    allele_points, chm13_points, matched = [], [], 0
    for hit in sorted(taken, key=lambda h: h["qs"]):
        a, c = block_points(hit["cigar"], hit["qs"], target_start + hit["ts"])
        allele_points += a
        chm13_points += c
        matched += sum(int(n) for n, op in CIGAR_OP.findall(hit["cigar"]) if op == "=")
    return np.array(allele_points), np.array(chm13_points), matched, len(taken)


def build_reference(plan_file, loci_file, slot, chm13_fasta, out_dir):
    out_dir = Path(out_dir)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(plan_file, sep="\t")
    plan = plan[plan["slot"] == slot]
    chm13 = pysam.FastaFile(chm13_fasta)
    segments, points = [], {}
    for row in plan.itertuples():
        allele = pysam.FastaFile(row.allele_fasta).fetch(row.allele_name).upper()
        target_start = max(0, row.region_start - ALIGNMENT_SLACK_BP)
        target_end = min(chm13.get_reference_length(row.chrom), row.region_end + ALIGNMENT_SLACK_BP)
        target = chm13.fetch(row.chrom, target_start, target_end).upper()
        allele_points, chm13_points, matched, n_alignments = align_allele(allele, target, target_start, tmp_dir)
        ends = interpolate(np.array([0, len(allele) - 1]), allele_points, chm13_points)
        chm13_start = int(np.clip(np.floor(ends[0]), target_start, target_end - 1))
        chm13_end = int(np.clip(np.floor(ends[1]) + 1, chm13_start + 1, target_end))
        segments.append({"chrom": row.chrom, "locus_id": row.locus_id, "chm13_start": chm13_start,
                         "chm13_end": chm13_end, "allele_name": row.allele_name, "allele_length": len(allele),
                         "matched_bp": matched, "n_alignments": n_alignments, "sequence": allele})
        points[f"allele_{row.locus_id}"] = allele_points
        points[f"chm13_{row.locus_id}"] = chm13_points
    columns = ["chrom", "locus_id", "chm13_start", "chm13_end", "allele_name", "allele_length", "matched_bp",
               "n_alignments", "sequence"]
    segments = pd.DataFrame(segments, columns=columns).sort_values(["chrom", "chm13_start"]).reset_index(drop=True)
    for chrom, chrom_segments in segments.groupby("chrom"):
        assert (chrom_segments["chm13_start"].to_numpy()[1:] >= chrom_segments["chm13_end"].to_numpy()[:-1]).all(), \
            f"swapped intervals overlap on {chrom}"

    # Each chromosome is rebuilt from left to right; a swap shifts every later reference position.
    segments["ref_start"], segments["ref_end"] = 0, 0
    with open(out_dir / "reference.fa", "w") as fasta, open(out_dir / "chrom.sizes", "w") as sizes:
        for chrom in chm13.references:
            sequence = chm13.fetch(chrom)
            pieces, cursor, shift = [], 0, 0
            for index, segment in segments[segments["chrom"] == chrom].iterrows():
                pieces += [sequence[cursor:segment.chm13_start], segment.sequence]
                segments.loc[index, "ref_start"] = segment.chm13_start + shift
                segments.loc[index, "ref_end"] = segment.chm13_start + shift + segment.allele_length
                shift += segment.allele_length - (segment.chm13_end - segment.chm13_start)
                cursor = segment.chm13_end
            pieces.append(sequence[cursor:])
            new_sequence = "".join(pieces)
            fasta.write(f">{chrom}\n{new_sequence}\n")
            sizes.write(f"{chrom}\t{len(new_sequence)}\n")

    segments = segments.drop(columns="sequence")
    segments.to_csv(out_dir / "segments.tsv", sep="\t", index=False)
    np.savez(out_dir / "points.npz", **points)

    lift = ReferenceLift(segments, points)
    loci = pd.read_csv(loci_file, sep="\t")
    loci = loci[loci["used"]]
    windows = [pd.DataFrame(columns=["chrom", "start", "end", "locus_id"])]
    for chrom, chrom_loci in loci.groupby("chrom"):
        windows.append(pd.DataFrame({"chrom": chrom, "start": lift.to_reference(chrom, chrom_loci["window_start"].to_numpy()),
                                     "end": lift.to_reference(chrom, chrom_loci["window_end"].to_numpy()),
                                     "locus_id": chrom_loci["locus_id"]}))
    pd.concat(windows).to_csv(out_dir / "windows.bed", sep="\t", index=False, header=False)
    for path in tmp_dir.iterdir():
        path.unlink()
    tmp_dir.rmdir()
    print(f"{slot}: {len(segments)} loci swapped", flush=True)
