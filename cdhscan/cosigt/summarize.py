"""Proxy haplotypes of every person at their CDH loci, from the cosigt results.

cosigt picks, per person and region, the pair of panel haplotypes whose graph node coverage best
matches the person's WGS reads (cosine similarity). Per person this module writes
  proxy_haplotypes.tsv   locus_id, region, proxy_haplotype_1, proxy_haplotype_2, cosine_similarity
                         (only the loci called as CDH in that person)
  proxy_haplotypes.fa    the two proxy allele sequences of every locus, named
                         "<locus_id>|proxy<k>|<haplotype name>"
"""

from pathlib import Path

import pandas as pd
import pysam


def summarize_person(sample, calls_file, cosigt_results_dir, alleles_dir, out_table, out_fasta):
    calls = pd.read_csv(calls_file, sep="\t")
    calls = calls[calls["sample"] == sample]
    rows = []
    with open(out_fasta, "w") as fasta:
        for call in calls.itertuples():
            chrom = call.region.split("_")[0]
            genotype_file = (Path(cosigt_results_dir) / "cosigt" / sample / chrom / call.region
                             / f"{call.region}.cosigt_genotype.tsv")
            genotype = pd.read_csv(genotype_file, sep="\t").iloc[0]
            proxies = [genotype["haplotype.1"], genotype["haplotype.2"]]
            rows.append({"locus_id": call.locus_id, "region": call.region, "proxy_haplotype_1": proxies[0],
                         "proxy_haplotype_2": proxies[1], "cosine_similarity": genotype["cosine.similarity"]})
            alleles = pysam.FastaFile(str(Path(alleles_dir) / f"{call.region}.fa"))
            for k, name in enumerate(proxies, start=1):
                fasta.write(f">{call.locus_id}|proxy{k}|{name}\n{alleles.fetch(name)}\n")
    pd.DataFrame(rows, columns=["locus_id", "region", "proxy_haplotype_1", "proxy_haplotype_2",
                                "cosine_similarity"]).to_csv(out_table, sep="\t", index=False, float_format="%.4f")
    print(f"{sample}: proxy haplotypes at {len(rows)} loci", flush=True)
