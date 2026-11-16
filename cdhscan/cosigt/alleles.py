"""One allele FASTA per region for cosigt, from the projected panel haplotypes.

Per region: the alleles of every panel haplotype that projects (cosigt.project), named
"<contig>:<start>-<end>"; HPRC contig names are already PanSN ("<sample>#<haplotype>#<contig>"),
as cosigt requires. Plus the GRCh38 allele
named "grch38#1#<chrom>:<start>-<end>" (cosigt needs the reference path in every graph). Each FASTA
gets a .fai index next to it, which cosigt requires.
"""

from pathlib import Path

import pandas as pd
import pysam

REFERENCE_PANSN = "grch38#1#"


def read_alleles(path):
    """(region, name, sequence) records of a projection FASTA whose headers are "<region>|<name>"."""
    with open(path) as handle:
        for header, sequence in zip(handle, handle):
            region, name = header[1:].rstrip("\n").split("|")
            yield region, name, sequence.rstrip("\n")


def collect_alleles(regions_file, haplotype_fastas, grch38_fasta, out_dir):
    """haplotype_fastas: the projection FASTAs of the panel haplotypes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = pd.read_csv(regions_file, sep="\t")
    for region in regions["region"]:
        (out_dir / f"{region}.fa").write_text("")
    # Haplotypes are streamed one at a time and appended to the region files, so the panel is never
    # held in memory.
    for path in haplotype_fastas:
        for region, name, sequence in read_alleles(path):
            with open(out_dir / f"{region}.fa", "a") as handle:
                handle.write(f">{name}\n{sequence}\n")
    grch38 = pysam.FastaFile(grch38_fasta)
    for row in regions.itertuples():
        sequence = grch38.fetch(row.chrom, row.grch38_start, row.grch38_end)
        with open(out_dir / f"{row.region}.fa", "a") as handle:
            handle.write(f">{REFERENCE_PANSN}{row.chrom}:{row.grch38_start}-{row.grch38_end}\n{sequence}\n")
        pysam.faidx(str(out_dir / f"{row.region}.fa"))
