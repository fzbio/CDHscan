"""Which CDH loci of a person are reconstructed, and which proxy allele goes into which reference.

Each person gets two individualized references: CHM13 with the replacement region (locus +-50 kb)
of every reconstructed locus swapped for cosigt's proxy haplotype 1 ("proxy1") or 2 ("proxy2").
Loci whose regions, extended by ALIGNMENT_SLACK_BP on both sides, overlap cannot both be swapped
in one reference: walking each chromosome from left to right, a locus is used only when its
extended region does not overlap the last used one.

Outputs in <out-dir>:
  loci.tsv           the person's loci with proxy haplotypes, used flag and the locus it overlaps
  plan.tsv           per slot and used locus: allele FASTA and allele name
  chm13_windows.bed  the 1 Mb windows of the used loci on CHM13 (contact filter)
"""

from pathlib import Path

import pandas as pd

from ..constants import ALIGNMENT_SLACK_BP, PROXY_SLOTS

LOCUS_COLUMNS = ["locus_id", "chrom", "start", "end", "window_start", "window_end", "region_start", "region_end"]


def choose_used_loci(loci):
    """Flag the loci whose slack-extended regions do not overlap a previously used locus."""
    loci = loci.sort_values(["chrom", "region_start"]).copy()
    loci["used"] = True
    loci["overlaps_locus"] = "."
    last_end, last_locus = {}, {}
    for index, locus in loci.iterrows():
        if locus.region_start - ALIGNMENT_SLACK_BP < last_end.get(locus.chrom, -1):
            loci.loc[index, ["used", "overlaps_locus"]] = [False, last_locus[locus.chrom]]
            continue
        last_end[locus.chrom] = locus.region_end + ALIGNMENT_SLACK_BP
        last_locus[locus.chrom] = locus.locus_id
    return loci


def plan_person(proxy_file, catalogue_file, alleles_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proxies = pd.read_csv(proxy_file, sep="\t")
    catalogue = pd.read_csv(catalogue_file, sep="\t")[LOCUS_COLUMNS]
    loci = choose_used_loci(proxies.merge(catalogue, on="locus_id", validate="one_to_one"))
    loci.to_csv(out_dir / "loci.tsv", sep="\t", index=False)
    used = loci[loci["used"]]
    used[["chrom", "window_start", "window_end", "locus_id"]].to_csv(out_dir / "chm13_windows.bed", sep="\t",
                                                                    index=False, header=False)
    rows = []
    for locus in used.itertuples():
        fasta = (Path(alleles_dir) / f"{locus.region}.fa").resolve()
        for slot, name in zip(PROXY_SLOTS, [locus.proxy_haplotype_1, locus.proxy_haplotype_2]):
            rows.append({"slot": slot, "locus_id": locus.locus_id, "chrom": locus.chrom,
                         "region_start": locus.region_start, "region_end": locus.region_end,
                         "allele_fasta": str(fasta), "allele_name": name})
    pd.DataFrame(rows, columns=["slot", "locus_id", "chrom", "region_start", "region_end", "allele_fasta",
                                "allele_name"]).to_csv(out_dir / "plan.tsv", sep="\t", index=False)
    print(f"{len(used)} of {len(loci)} loci reconstructed; left out (overlap): "
          f"{', '.join(loci.loc[~loci['used'], 'locus_id']) or 'none'}", flush=True)
