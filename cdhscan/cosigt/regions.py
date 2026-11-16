"""The GRCh38 regions cosigt genotypes: the replacement regions of the loci called as CDH.

A locus called as CDH in any person (score.score_person, column cdh), or listed as an extra locus
(genotyped in every person), gives one region: the locus
+-50 kb on CHM13 lifted to GRCh38 (catalogue columns grch38_start, grch38_end), kept when at least
MIN_ALIGNED_FRACTION of it aligns to GRCh38. cosigt names a region chrom_start_end (it reads the
chromosome from the name).

Outputs:
  regions.tsv   locus_id, chrom, grch38_start, grch38_end, region (one row per region)
  regions.bed   chrom, grch38_start, grch38_end, region (for the projection of the haplotypes)
  calls.tsv     sample, locus_id, region: the people whose proxy haplotypes are reported per region
"""

from pathlib import Path

import pandas as pd

from ..constants import MIN_ALIGNED_FRACTION


def region_name(chrom, start, end):
    return f"{chrom}_{start}_{end}"


def select_regions(catalogue_file, prediction_files, out_dir, extra_loci=(), include_calls=True):
    """prediction_files: sample -> predictions.tsv of that person; extra_loci: catalogue locus ids;
    include_calls = False genotypes the extra loci only."""
    catalogue = pd.read_csv(catalogue_file, sep="\t").set_index("locus_id")
    calls = []
    for sample, path in prediction_files.items():
        predictions = pd.read_csv(path, sep="\t")
        called = set(predictions.loc[predictions["cdh"], "locus_id"]) if include_calls else set()
        called |= set(extra_loci)
        calls.append(pd.DataFrame({"sample": sample, "locus_id": sorted(called)}))
    calls = pd.concat(calls, ignore_index=True)
    regions = catalogue.loc[calls["locus_id"].unique(), ["chrom", "grch38_start", "grch38_end",
                                                         "region_aligned_fraction"]].reset_index()
    regions = regions[regions["region_aligned_fraction"] >= MIN_ALIGNED_FRACTION].drop(columns="region_aligned_fraction")
    regions["region"] = [region_name(*row) for row in regions[["chrom", "grch38_start", "grch38_end"]].itertuples(index=False)]
    regions = regions.sort_values(["chrom", "grch38_start"]).drop_duplicates("region")
    calls = calls.merge(regions[["locus_id", "region"]], on="locus_id")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions.to_csv(out_dir / "regions.tsv", sep="\t", index=False)
    regions[["chrom", "grch38_start", "grch38_end", "region"]].to_csv(out_dir / "regions.bed", sep="\t",
                                                                     index=False, header=False)
    calls.to_csv(out_dir / "calls.tsv", sep="\t", index=False)
    print(f"{len(regions)} regions from {len(calls)} CDH calls of {calls['sample'].nunique()} people", flush=True)
