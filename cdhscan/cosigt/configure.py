"""Write the cosigt configuration into the cosigt pipeline copy.

cosigt runs in its "custom alleles" mode: the alleles of each region are the panel FASTAs of
cosigt.alleles, so cosigt neither aligns assemblies nor projects regions itself. The people are
genotyped from their GRCh38 WGS alignments. All paths are absolute, because cosigt runs from its
own folder and binds them into its containers.

Changed against cosigt's config.yaml.example: output, reference, allele_source (custom), the two
tmpdir entries, and the odgi / pangene plots (off). Everything else, including the read mode
(short reads, bwa-mem2 and k-mer rescue of unmapped reads) and all resources, stays at the cosigt
defaults.

Writes in <pipeline-dir>/cosigt_smk/config: config.yaml, samples.tsv, regions.bed, alleles.tsv
"""

import os
from pathlib import Path

import pandas as pd


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one '{old.strip()}' in the cosigt example config"
    return text.replace(old, new)


def configure_cosigt(alignments, regions_file, alleles_dir, reference, pipeline_dir, results_dir, tmp_dir):
    """alignments: sample -> GRCh38 CRAM or BAM of that person."""
    config_dir = Path(pipeline_dir) / "cosigt_smk" / "config"
    pd.DataFrame({"sample": list(alignments), "alignment": [os.path.realpath(path) for path in alignments.values()]}).to_csv(
        config_dir / "samples.tsv", sep="\t", index=False)
    regions = pd.read_csv(regions_file, sep="\t")
    regions[["chrom", "grch38_start", "grch38_end", "region"]].to_csv(config_dir / "regions.bed", sep="\t",
                                                                     index=False, header=False)
    fastas = [os.path.realpath(Path(alleles_dir) / f"{region}.fa") for region in regions["region"]]
    pd.DataFrame({"region": regions["region"], "fasta": fastas}).to_csv(config_dir / "alleles.tsv", sep="\t", index=False)

    results_dir, tmp_dir = os.path.realpath(results_dir), os.path.realpath(tmp_dir)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    config = (config_dir / "config.yaml.example").read_text()
    config = replace_once(config, "\noutput: results\n", f"\noutput: {results_dir}\n")
    config = replace_once(config, "\nreference: /path/to/reference.fa.gz\n", f"\nreference: {os.path.realpath(reference)}\n")
    config = replace_once(config, "\nallele_source: assemblies\n", "\nallele_source: custom\n")
    config = replace_once(config, "\ntmpdir: /tmp\n", f"\ntmpdir: {tmp_dir}\n")
    config = replace_once(config, "\n  tmpdir: /tmp\n", f"\n  tmpdir: {tmp_dir}\n")
    config = replace_once(config, "\nodgi_viz: true\n", "\nodgi_viz: false\n")
    config = replace_once(config, "\npangene_viz: true\n", "\npangene_viz: false\n")
    (config_dir / "config.yaml").write_text(config)
    print(f"cosigt configured: {len(alignments)} people, {len(regions)} regions", flush=True)
