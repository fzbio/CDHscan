# Prediction: per person, Hi-C bands from the CHM13 .hic, WGS read counts from the GRCh38
# alignment, then the CDH probability of every catalogue locus with the released model.

SAMPLES = read_samples(config["samples"]) if config.get("samples") else None
MODEL_DIR = Path(config["model_dir"])


rule predict:
    input:
        expand(OUT / "{sample}" / "predictions.tsv", sample=SAMPLES.index if SAMPLES is not None else [])


rule extract_hic:
    output:
        directory(WORK / "bands" / "{sample}")
    params:
        fetch=lambda w: hic_fetch_command(SAMPLES.loc[w.sample, "hic"], WORK / "hic" / f"{w.sample}.hic"),
        hic=lambda w: WORK / "hic" / f"{w.sample}.hic",
        chroms=" ".join(CHROMS)
    log:
        OUT / "logs" / "extract_hic" / "{sample}.log"
    resources:
        mem_mb=8000
    shell:
        """
        mkdir -p $(dirname {params.hic})
        {params.fetch}
        cdhscan extract-hic --hic {params.hic} --chroms {params.chroms} --out-dir {output} > {log} 2>&1
        rm -f {params.hic}
        """


rule count_reads:
    input:
        alignment=lambda w: SAMPLES.loc[w.sample, "alignment"],
        bins=MODEL_DIR / "bins.tsv"
    output:
        WORK / "counts" / "{sample}.npy"
    params:
        reference=f"--reference {config['grch38_fasta']}" if config.get("grch38_fasta") else "",
        chroms=" ".join(CHROMS),
        region=region_option()
    log:
        OUT / "logs" / "count_reads" / "{sample}.log"
    resources:
        mem_mb=8000
    shell:
        "cdhscan count-reads --alignment {input.alignment} {params.reference} --bins {input.bins} "
        "--chroms {params.chroms} {params.region} --out {output} > {log} 2>&1"


rule score:
    input:
        bands=WORK / "bands" / "{sample}",
        counts=WORK / "counts" / "{sample}.npy",
        model=MODEL_DIR / "MD5SUMS"
    output:
        OUT / "{sample}" / "predictions.tsv"
    params:
        model_dir=MODEL_DIR,
        chroms=" ".join(CHROMS),
        region=region_option()
    threads: config.get("threads", 4)
    log:
        OUT / "logs" / "score" / "{sample}.log"
    resources:
        mem_mb=16000
    shell:
        "cdhscan score --model-dir {params.model_dir} --bands {input.bands} --counts {input.counts} "
        "--chroms {params.chroms} {params.region} --threads {threads} --out {output} > {log} 2>&1"
