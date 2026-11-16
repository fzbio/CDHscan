# cosigt: proxy haplotypes at the loci called as CDH.
#
#   1. regions: the replacement regions (locus +-50 kb, lifted to GRCh38) of every CDH call and
#      of cosigt.extra_loci
#   2. per HPRC release 2 haplotype: its allele at every region, read from the public
#      assembly-to-GRCh38 alignment and assembly (S3, remote reads; only indexes are downloaded).
#      The people of the sample sheet and cosigt.exclude_panel_samples are never in the panel.
#   3. one allele FASTA per region (panel alleles + GRCh38 allele)
#   4. cosigt (pinned release) genotypes every person at every region from their WGS reads
#   5. per person: the two proxy haplotypes of each of their CDH loci, and their sequences

COSIGT = config["cosigt"]
COSIGT_WORK = WORK / "cosigt"
COSIGT_PIPELINE = COSIGT_WORK / "pipeline"


def panel_haplotypes():
    haplotypes = pd.read_csv(COSIGT["haplotypes"], sep="\t")
    excluded = set(COSIGT.get("exclude_panel_samples") or []) | set(SAMPLES.index)
    haplotypes = haplotypes[~haplotypes["sample"].isin(excluded)]
    haplotypes["haplotype_id"] = haplotypes["sample"] + "_hap" + haplotypes["haplotype"].astype(str)
    return haplotypes.set_index("haplotype_id")


HAPLOTYPES = panel_haplotypes() if SAMPLES is not None else None


def haplotype_file(column):
    return lambda wildcards: HAPLOTYPES.loc[wildcards.haplotype_id, column]


def projected_fastas(wildcards):
    """Projection FASTAs of the panel haplotypes; a prepared folder replaces step 2 when given (tests)."""
    if COSIGT.get("projected_haplotypes_dir"):
        return sorted(str(path) for path in Path(COSIGT["projected_haplotypes_dir"]).glob("*.fa"))
    return [str(COSIGT_WORK / "haplotypes" / f"{haplotype_id}.fa") for haplotype_id in HAPLOTYPES.index]


rule cosigt:
    input:
        expand(OUT / "{sample}" / "proxy_haplotypes.tsv", sample=SAMPLES.index if SAMPLES is not None else [])


rule cosigt_regions:
    input:
        predictions=expand(OUT / "{sample}" / "predictions.tsv", sample=SAMPLES.index),
        catalogue=MODEL_DIR / "catalogue.tsv"
    output:
        regions=COSIGT_WORK / "regions.tsv",
        bed=COSIGT_WORK / "regions.bed",
        calls=COSIGT_WORK / "calls.tsv"
    params:
        pairs=" ".join(f"{s}={OUT / s / 'predictions.tsv'}" for s in SAMPLES.index) if SAMPLES is not None else "",
        extra=" ".join(COSIGT.get("extra_loci") or []),
        extra_only="" if COSIGT.get("include_calls", True) else "--extra-only"
    shell:
        "cdhscan cosigt-regions --catalogue {input.catalogue} --predictions {params.pairs} "
        "--extra-loci {params.extra} {params.extra_only} --out-dir $(dirname {output.regions})"


rule project_haplotype:
    input:
        regions=COSIGT_WORK / "regions.bed"
    output:
        table=COSIGT_WORK / "haplotypes" / "{haplotype_id}.tsv",
        fasta=COSIGT_WORK / "haplotypes" / "{haplotype_id}.fa"
    params:
        bam=haplotype_file("grch38_bam_url"),
        assembly=haplotype_file("assembly_url"),
        fai=haplotype_file("fai_url"),
        gzi=haplotype_file("gzi_url"),
        tmp=lambda w: COSIGT_WORK / "haplotypes" / f"{w.haplotype_id}_index"
    log:
        OUT / "logs" / "project_haplotype" / "{haplotype_id}.log"
    retries: 2
    resources:
        mem_mb=4000
    shell:
        """
        mkdir -p {params.tmp}
        curl -sSfL --retry 5 -o {params.tmp}/alignment.bai {params.bam}.bai
        curl -sSfL --retry 5 -o {params.tmp}/assembly.fai {params.fai}
        curl -sSfL --retry 5 -o {params.tmp}/assembly.gzi {params.gzi}
        cdhscan project-haplotype --bam {params.bam} --bai {params.tmp}/alignment.bai \
            --assembly {params.assembly} --fai {params.tmp}/assembly.fai --gzi {params.tmp}/assembly.gzi \
            --regions {input.regions} --out-table {output.table} --out-fasta {output.fasta} > {log} 2>&1
        rm -rf {params.tmp}
        """


rule collect_alleles:
    input:
        regions=COSIGT_WORK / "regions.tsv",
        fastas=projected_fastas
    output:
        directory(COSIGT_WORK / "alleles")
    params:
        grch38=config["grch38_fasta"]
    shell:
        "cdhscan collect-alleles --regions {input.regions} --fastas {input.fastas} --grch38 {params.grch38} "
        "--out-dir {output}"


rule configure_cosigt:
    input:
        regions=COSIGT_WORK / "regions.tsv",
        alleles=COSIGT_WORK / "alleles"
    output:
        COSIGT_WORK / "configured"
    params:
        commit=COSIGT["commit"],
        pipeline=COSIGT_PIPELINE,
        alignments=" ".join(f"{s}={SAMPLES.loc[s, 'alignment']}" for s in SAMPLES.index) if SAMPLES is not None else "",
        reference=config["grch38_fasta"],
        bwa_mem2=COSIGT.get("bwa_mem2_binary") or "",
        make_options=f"PROFILE={COSIGT['profile']} SOFTWARE={COSIGT['software']} TARGET=cosigt CORES={COSIGT['cores']}"
    log:
        OUT / "logs" / "configure_cosigt.log"
    shell:
        """
        if [[ ! -d {params.pipeline}/.git ]]; then
            git clone --quiet https://github.com/davidebolo1993/cosigt {params.pipeline}
        fi
        git -C {params.pipeline} checkout --quiet {params.commit}
        # Optional fixed bwa-mem2 build instead of the launcher's automatic choice (cosigt.bwa_mem2_binary).
        RULES={params.pipeline}/cosigt_smk/workflow/rules/bwa-mem2.smk
        git -C {params.pipeline} checkout --quiet -- cosigt_smk/workflow/rules/bwa-mem2.smk
        if [[ -n "{params.bwa_mem2}" ]]; then
            sed -i "s#bwa-mem2 index -p#{params.bwa_mem2} index -p#; s#| bwa-mem2 mem #| {params.bwa_mem2} mem #" $RULES
            [[ $(grep -c "{params.bwa_mem2}" $RULES) == 2 ]]
        fi
        cdhscan configure-cosigt --alignments {params.alignments} --regions {input.regions} \
            --alleles-dir {input.alleles} --reference {params.reference} --pipeline-dir {params.pipeline} \
            --results-dir {COSIGT_WORK}/results --tmp-dir {COSIGT_WORK}/tmp > {log} 2>&1
        make -C {params.pipeline} check {params.make_options} >> {log} 2>&1
        touch {output}
        """


rule run_cosigt:
    input:
        COSIGT_WORK / "configured"
    output:
        COSIGT_WORK / "finished"
    params:
        pipeline=COSIGT_PIPELINE,
        make_options=f"PROFILE={COSIGT['profile']} SOFTWARE={COSIGT['software']} TARGET=cosigt CORES={COSIGT['cores']}"
    log:
        OUT / "logs" / "run_cosigt.log"
    shell:
        """
        make -C {params.pipeline} run {params.make_options} > {log} 2>&1
        touch {output}
        """


rule summarize_cosigt:
    input:
        calls=COSIGT_WORK / "calls.tsv",
        finished=COSIGT_WORK / "finished",
        alleles=COSIGT_WORK / "alleles"
    output:
        table=OUT / "{sample}" / "proxy_haplotypes.tsv",
        fasta=OUT / "{sample}" / "proxy_haplotypes.fa"
    shell:
        "cdhscan summarize-cosigt --sample {wildcards.sample} --calls {input.calls} --results-dir {COSIGT_WORK}/results "
        "--alleles-dir {input.alleles} --out-table {output.table} --out-fasta {output.fasta}"
