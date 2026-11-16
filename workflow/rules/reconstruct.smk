# Reconstruction: the person's Hi-C reads remapped to two individualized references.
#
#   1. plan: the person's CDH loci with proxy haplotypes (cosigt target); loci whose regions
#      (+-20 kb) overlap are not swapped together
#   2. two references per person: CHM13 with every used replacement region (locus +-50 kb)
#      swapped for proxy haplotype 1 or 2 (allele aligned to CHM13 with minimap2; the alignment
#      also converts positions back to CHM13); BWA index of each
#   3. every Hi-C read pair file mapped to CHM13 as Ni et al. map Hi-C reads (bwa mem -5SP -T0;
#      pairtools parse: MAPQ >= 40, 5'-most unique alignments of a walk); the window contacts are
#      kept, and every read pair that could form a window contact on an individualized reference
#      (all pairs except those with an end mapped uniquely outside every window) goes on to step 4
#   4. those read pairs mapped to each individualized reference independently, same settings
#   5. duplicates removed per library (pairtools dedup)
#   6. window maps on CHM13 and on both proxy references (10 kb CHM13 bins of the 1 Mb window),
#      haplotype-specific contacts (read pair found on one proxy reference only)

RECON = config.get("reconstruct", {})
RECON_WORK = WORK / "reconstruct"


def read_hic_reads(path):
    reads = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    assert {"sample", "library", "r1", "r2"} <= set(reads.columns), f"{path}: needs columns sample, library, r1, r2"
    # One file id per read pair file: library and its position among the person's files.
    reads["file_id"] = reads["library"] + "." + reads.groupby("sample").cumcount().astype(str)
    return reads.set_index(["sample", "file_id"], drop=False)


HIC_READS = read_hic_reads(RECON["hic_reads"]) if RECON.get("hic_reads") else None
RECON_SAMPLES = sorted(set(HIC_READS["sample"])) if HIC_READS is not None else []
CHM13_INDEX = RECON.get("chm13_bwa_index") or str(RECON_WORK / "chm13" / "index")


def read_files(sample):
    return list(HIC_READS.loc[HIC_READS["sample"] == sample, "file_id"]) if HIC_READS is not None else []


def stage_reads_command(source, local_file):
    """Shell commands that make a FASTQ (local path or http(s)/s3 URL) available as local_file."""
    if source.startswith("s3://"):
        return f"aws s3 cp --no-sign-request --only-show-errors {source} {local_file}"
    if source.startswith(("http://", "https://")):
        return f"curl -sSfL --retry 5 -o {local_file} {source}"
    return f"ln -sf $(realpath {source}) {local_file}"


def staged_reads(wildcards, mate):
    row = HIC_READS.loc[(wildcards.sample, wildcards.file)]
    local = RECON_WORK / wildcards.sample / "staged" / f"{wildcards.file}_{mate}.fq.gz"
    return stage_reads_command(row[mate.lower()], local)


rule reconstruct:
    input:
        expand(OUT / "{sample}" / "reconstruction" / "maps.npz", sample=RECON_SAMPLES)


rule reconstruct_plan:
    input:
        proxies=OUT / "{sample}" / "proxy_haplotypes.tsv",
        alleles=COSIGT_WORK / "alleles",
        catalogue=MODEL_DIR / "catalogue.tsv"
    output:
        loci=RECON_WORK / "{sample}" / "loci.tsv",
        plan=RECON_WORK / "{sample}" / "plan.tsv",
        windows=RECON_WORK / "{sample}" / "chm13_windows.bed"
    shell:
        "cdhscan reconstruct-plan --proxies {input.proxies} --catalogue {input.catalogue} "
        "--alleles-dir {input.alleles} --out-dir $(dirname {output.loci})"


rule chm13_index:
    output:
        RECON_WORK / "chm13" / "index.sa"
    params:
        fasta=RECON.get("chm13_fasta"),
        prefix=RECON_WORK / "chm13" / "index"
    log:
        OUT / "logs" / "chm13_index.log"
    resources:
        mem_mb=8000
    shell:
        "bwa index -p {params.prefix} {params.fasta} > {log} 2>&1"


rule chm13_sizes:
    output:
        RECON_WORK / "chm13" / "chrom.sizes"
    params:
        fasta=RECON.get("chm13_fasta")
    shell:
        "cut -f1,2 {params.fasta}.fai > {output}"


rule reconstruct_reference:
    input:
        plan=RECON_WORK / "{sample}" / "plan.tsv",
        loci=RECON_WORK / "{sample}" / "loci.tsv"
    output:
        index=RECON_WORK / "{sample}" / "{slot}" / "index.sa",
        sizes=RECON_WORK / "{sample}" / "{slot}" / "chrom.sizes",
        segments=RECON_WORK / "{sample}" / "{slot}" / "segments.tsv",
        points=RECON_WORK / "{sample}" / "{slot}" / "points.npz",
        windows=RECON_WORK / "{sample}" / "{slot}" / "windows.bed"
    params:
        chm13=RECON.get("chm13_fasta"),
        out_dir=lambda w: RECON_WORK / w.sample / w.slot
    log:
        OUT / "logs" / "reconstruct_reference" / "{sample}_{slot}.log"
    resources:
        mem_mb=16000
    shell:
        """
        cdhscan build-reference --plan {input.plan} --loci {input.loci} --slot {wildcards.slot} \
            --chm13 {params.chm13} --out-dir {params.out_dir} > {log} 2>&1
        bwa index -p {params.out_dir}/index {params.out_dir}/reference.fa >> {log} 2>&1
        rm {params.out_dir}/reference.fa
        """


rule reconstruct_map_chm13:
    input:
        windows=RECON_WORK / "{sample}" / "chm13_windows.bed",
        index=CHM13_INDEX + ".sa",
        sizes=RECON_WORK / "chm13" / "chrom.sizes"
    output:
        pairs=RECON_WORK / "{sample}" / "pairs" / "chm13" / "{file}.pairs.gz",
        r1=RECON_WORK / "{sample}" / "stage2" / "{file}_R1.fq.gz",
        r2=RECON_WORK / "{sample}" / "stage2" / "{file}_R2.fq.gz"
    params:
        stage_r1=lambda w: staged_reads(w, "R1"),
        stage_r2=lambda w: staged_reads(w, "R2"),
        staged=lambda w: RECON_WORK / w.sample / "staged" / w.file,
        tmp=lambda w: RECON_WORK / w.sample / "tmp" / f"chm13_{w.file}",
        index=CHM13_INDEX
    threads: RECON.get("threads", 8)
    log:
        OUT / "logs" / "reconstruct_map" / "{sample}_chm13_{file}.log"
    resources:
        mem_mb=16000
    shell:
        """
        mkdir -p $(dirname {params.staged}) {params.tmp}
        {params.stage_r1}
        {params.stage_r2}
        bwa mem -5SP -T0 -t {threads} {params.index} {params.staged}_R1.fq.gz {params.staged}_R2.fq.gz 2> {log} \
            | pairtools parse --min-mapq 40 --walks-policy 5unique --max-inter-align-gap 30 --drop-sam \
                  --chroms-path {input.sizes} 2>> {log} \
            | cdhscan window-pairs --windows {input.windows} --stage2-ids {params.tmp}/ids.txt 2>> {log} \
            | pairtools sort --nproc 2 --memory 4G --tmpdir {params.tmp} -o {output.pairs}
        seqtk subseq {params.staged}_R1.fq.gz {params.tmp}/ids.txt | gzip -1 > {output.r1}
        seqtk subseq {params.staged}_R2.fq.gz {params.tmp}/ids.txt | gzip -1 > {output.r2}
        rm -rf {params.tmp} {params.staged}_R1.fq.gz {params.staged}_R2.fq.gz
        """


rule reconstruct_map_proxy:
    input:
        r1=RECON_WORK / "{sample}" / "stage2" / "{file}_R1.fq.gz",
        r2=RECON_WORK / "{sample}" / "stage2" / "{file}_R2.fq.gz",
        index=RECON_WORK / "{sample}" / "{slot}" / "index.sa",
        sizes=RECON_WORK / "{sample}" / "{slot}" / "chrom.sizes",
        windows=RECON_WORK / "{sample}" / "{slot}" / "windows.bed"
    output:
        RECON_WORK / "{sample}" / "pairs" / "{slot}" / "{file}.pairs.gz"
    params:
        index=lambda w: RECON_WORK / w.sample / w.slot / "index",
        tmp=lambda w: RECON_WORK / w.sample / "tmp" / f"{w.slot}_{w.file}"
    threads: RECON.get("threads", 8)
    log:
        OUT / "logs" / "reconstruct_map" / "{sample}_{slot}_{file}.log"
    resources:
        mem_mb=16000
    shell:
        """
        mkdir -p {params.tmp}
        bwa mem -5SP -T0 -t {threads} {params.index} {input.r1} {input.r2} 2> {log} \
            | pairtools parse --min-mapq 40 --walks-policy 5unique --max-inter-align-gap 30 --drop-sam \
                  --chroms-path {input.sizes} 2>> {log} \
            | cdhscan window-pairs --windows {input.windows} 2>> {log} \
            | pairtools sort --nproc 2 --memory 4G --tmpdir {params.tmp} -o {output}
        rm -rf {params.tmp}
        """


def arm_pairs(wildcards):
    return [str(RECON_WORK / wildcards.sample / "pairs" / wildcards.arm / f"{file}.pairs.gz")
            for file in read_files(wildcards.sample)]


def library_merge_commands(wildcards, tmp, stats_prefix):
    """Per library: merge its files and mark duplicates; one shell line per library."""
    rows = HIC_READS[HIC_READS["sample"] == wildcards.sample]
    commands = []
    for library, files in rows.groupby("library")["file_id"]:
        inputs = " ".join(str(RECON_WORK / wildcards.sample / "pairs" / wildcards.arm / f"{f}.pairs.gz") for f in files)
        commands.append(f"pairtools merge --nproc 2 --memory 4G --tmpdir {tmp} {inputs} "
                        f"| pairtools dedup --mark-dups --output-stats {stats_prefix}.{library}.dedup_stats.txt "
                        f"-o {tmp}/{library}.pairs.gz")
    return "\n".join(commands)


rule reconstruct_dedup:
    input:
        arm_pairs
    output:
        OUT / "{sample}" / "reconstruction" / "{arm}.dedup.pairs.gz"
    params:
        tmp=lambda w: RECON_WORK / w.sample / "tmp" / f"dedup_{w.arm}",
        commands=lambda w: library_merge_commands(w, RECON_WORK / w.sample / "tmp" / f"dedup_{w.arm}",
                                                  OUT / w.sample / "reconstruction" / w.arm)
    resources:
        mem_mb=8000
    shell:
        """
        mkdir -p {params.tmp}
        {params.commands}
        pairtools merge --nproc 2 --memory 4G --tmpdir {params.tmp} {params.tmp}/*.pairs.gz -o {output}
        rm -rf {params.tmp}
        """


rule reconstruct_maps:
    input:
        loci=RECON_WORK / "{sample}" / "loci.tsv",
        pairs=expand(OUT / "{{sample}}" / "reconstruction" / "{arm}.dedup.pairs.gz", arm=MAP_ARMS),
        references=expand(RECON_WORK / "{{sample}}" / "{slot}" / "points.npz", slot=PROXY_SLOTS)
    output:
        maps=OUT / "{sample}" / "reconstruction" / "maps.npz",
        contacts=OUT / "{sample}" / "reconstruction" / "contacts.tsv",
        specific=OUT / "{sample}" / "reconstruction" / "specific.tsv",
        region=OUT / "{sample}" / "reconstruction" / "region.tsv"
    resources:
        mem_mb=16000
    shell:
        "cdhscan reconstruct-maps --loci {input.loci} --pairs-dir $(dirname {output.maps}) "
        "--references-dir {RECON_WORK}/{wildcards.sample} --out-dir $(dirname {output.maps})"
