# Training: rebuild the released model from public data.
#
#   1. per training person: Hi-C bands from the HPRC chm13.hic and WGS read counts from the
#      1000 Genomes 30x GRCh38 CRAM
#   2. 10 kb CHM13 bins lifted to GRCh38 (UCSC hs1ToHg38 chain)
#   3. locus catalogue from the CHM13 CDH labels of all training people (bedtools), training and
#      validation windows matched within each locus
#   4. Hi-C and WGS inputs of the windows (clean panel, leave-one-out for negatives)
#   5. the network, once per seed (early stopping on the validation people)
#   6. panel reference of every scored locus, and the model folder that the predict target reads

TRAIN = config.get("train", {})
TRAIN_OUT = Path(TRAIN.get("output_dir", OUT / "train"))
PARTS = [1, 2, 3, 4]


def training_people():
    people = pd.read_csv(TRAIN["people"], sep="\t", dtype=str)
    assert set(people["role"]) == {"train", "validation"}, "roles must be train or validation"
    return people.set_index("sample", drop=False)


PEOPLE = training_people() if TRAIN else None
PERSON_BANDS = [str(TRAIN_OUT / "bands" / person) for person in PEOPLE.index] if TRAIN else []
PERSON_COUNTS = [str(TRAIN_OUT / "counts" / f"{person}.npy") for person in PEOPLE.index] if TRAIN else []


rule train:
    input:
        TRAIN_OUT / "model" / "MD5SUMS"


rule train_extract_hic:
    output:
        directory(TRAIN_OUT / "bands" / "{person}")
    params:
        fetch=lambda w: hic_fetch_command(PEOPLE.loc[w.person, "hic"], TRAIN_OUT / "hic" / f"{w.person}.hic"),
        hic=lambda w: TRAIN_OUT / "hic" / f"{w.person}.hic",
        chroms=" ".join(LABELLED_CHROMS)
    log:
        TRAIN_OUT / "logs" / "extract_hic" / "{person}.log"
    resources:
        mem_mb=8000
    shell:
        """
        mkdir -p $(dirname {params.hic})
        {params.fetch}
        cdhscan extract-hic --hic {params.hic} --chroms {params.chroms} --out-dir {output} > {log} 2>&1
        rm -f {params.hic}
        """


rule train_chain:
    output:
        TRAIN_OUT / "hs1ToHg38.over.chain.gz"
    params:
        url=TRAIN.get("chain_url")
    shell:
        "curl -sSfL --retry 5 -o {output} {params.url}"


rule train_bins:
    input:
        chain=TRAIN_OUT / "hs1ToHg38.over.chain.gz",
        bands=PERSON_BANDS[:1]
    output:
        TRAIN_OUT / "bins.tsv"
    shell:
        "cdhscan bins --chain {input.chain} --band-dir {input.bands} --out {output}"


rule train_count_reads:
    input:
        alignment=lambda w: TRAIN["alignment_pattern"].format(sample=w.person),
        bins=TRAIN_OUT / "bins.tsv"
    output:
        TRAIN_OUT / "counts" / "{person}.npy"
    params:
        reference=config["grch38_fasta"]
    log:
        TRAIN_OUT / "logs" / "count_reads" / "{person}.log"
    resources:
        mem_mb=8000
    shell:
        "cdhscan count-reads --alignment {input.alignment} --reference {params.reference} --bins {input.bins} "
        "--out {output} > {log} 2>&1"


rule train_catalogue:
    # Loci: the training people's CDH regions of at most MAX_LOCUS_BP merged across people; merged
    # loci longer than that dropped. Carriers: every training person with a region of any length
    # overlapping the locus (label columns: chrom, start, end, cdh id, sample, ...).
    input:
        labels=TRAIN.get("labels", []),
        people=TRAIN.get("people", [])
    output:
        regions=TRAIN_OUT / "catalogue" / "regions.bed",
        loci=TRAIN_OUT / "catalogue" / "loci.bed"
    params:
        max_bp=1_000_000
    shell:
        """
        gzip -dcf {input.labels} | tail -n +2 \
            | awk -F'\\t' 'BEGIN {{OFS = "\\t"}} FNR == NR {{if (FNR > 1) panel[$1] = 1; next}} $5 in panel {{print $1, $2, $3, $5}}' \
                {input.people} - \
            | sort -k1,1 -k2,2n > {output.regions}
        awk -v max={params.max_bp} '$3 - $2 <= max' {output.regions} \
            | bedtools merge -i - \
            | awk -v max={params.max_bp} 'BEGIN {{OFS = "\\t"}} $3 - $2 <= max {{printf "%s\\t%s\\t%s\\tCDH_LOCUS_%05d\\n", $1, $2, $3, ++n}}' \
            | bedtools map -a - -b {output.regions} -c 4 -o distinct > {output.loci}
        """


rule train_windows:
    input:
        loci=TRAIN_OUT / "catalogue" / "loci.bed",
        labels=TRAIN.get("labels", []),
        people=TRAIN.get("people", []),
        chain=TRAIN_OUT / "hs1ToHg38.over.chain.gz",
        bands=PERSON_BANDS
    output:
        loci=TRAIN_OUT / "windows" / "loci.tsv",
        windows=TRAIN_OUT / "windows" / "windows.tsv"
    resources:
        mem_mb=16000
    shell:
        "cdhscan windows --loci-bed {input.loci} --labels {input.labels} --people {input.people} "
        "--band-root {TRAIN_OUT}/bands --chain {input.chain} --out-dir $(dirname {output.loci})"


rule train_hic_inputs:
    input:
        windows=TRAIN_OUT / "windows" / "windows.tsv",
        bands=PERSON_BANDS
    output:
        TRAIN_OUT / "hic_inputs" / "fit_{chrom}_part{part}.npy"
    params:
        people=TRAIN.get("people")
    resources:
        mem_mb=16000
    shell:
        "cdhscan hic-inputs --windows-dir $(dirname {input.windows}) --people {params.people} "
        "--band-root {TRAIN_OUT}/bands --chrom {wildcards.chrom} --out {output}"


rule train_wgs_tracks:
    input:
        windows=TRAIN_OUT / "windows" / "windows.tsv",
        bins=TRAIN_OUT / "bins.tsv",
        counts=PERSON_COUNTS
    output:
        TRAIN_OUT / "wgs_tracks.npy"
    params:
        people=TRAIN.get("people")
    resources:
        mem_mb=16000
    shell:
        "cdhscan wgs-tracks --windows-dir $(dirname {input.windows}) --people {params.people} "
        "--counts-dir {TRAIN_OUT}/counts --bins {input.bins} --out {output}"


rule train_model:
    input:
        windows=TRAIN_OUT / "windows" / "windows.tsv",
        hic=expand(TRAIN_OUT / "hic_inputs" / "fit_{chrom}_part{part}.npy", chrom=LABELLED_CHROMS, part=PARTS),
        tracks=TRAIN_OUT / "wgs_tracks.npy"
    output:
        TRAIN_OUT / "models" / "seed_{seed}" / "best_model.pt"
    threads: TRAIN.get("threads", 8)
    log:
        TRAIN_OUT / "logs" / "train" / "seed_{seed}.log"
    resources:
        mem_mb=128000
    shell:
        "cdhscan train --windows {input.windows} --hic-inputs-dir {TRAIN_OUT}/hic_inputs --tracks {input.tracks} "
        "--seed {wildcards.seed} --threads {threads} --out-dir $(dirname {output}) > {log} 2>&1"


rule train_panel_reference:
    input:
        loci=TRAIN_OUT / "windows" / "loci.tsv",
        bins=TRAIN_OUT / "bins.tsv",
        bands=PERSON_BANDS,
        counts=PERSON_COUNTS
    output:
        TRAIN_OUT / "panel_reference" / "{chrom}.npz"
    params:
        people=TRAIN.get("people")
    resources:
        mem_mb=24000
    shell:
        "cdhscan panel-reference --loci {input.loci} --people {params.people} --band-root {TRAIN_OUT}/bands "
        "--counts-dir {TRAIN_OUT}/counts --bins {input.bins} --chrom {wildcards.chrom} --out {output}"


rule train_package:
    input:
        loci=TRAIN_OUT / "windows" / "loci.tsv",
        bins=TRAIN_OUT / "bins.tsv",
        models=expand(TRAIN_OUT / "models" / "seed_{seed}" / "best_model.pt", seed=SEEDS),
        panel=expand(TRAIN_OUT / "panel_reference" / "{chrom}.npz", chrom=LABELLED_CHROMS)
    output:
        TRAIN_OUT / "model" / "MD5SUMS"
    params:
        seed_dirs=" ".join(str(TRAIN_OUT / "models" / f"seed_{seed}") for seed in SEEDS)
    shell:
        "cdhscan package --loci {input.loci} --bins {input.bins} --seed-dirs {params.seed_dirs} "
        "--panel-reference-dir {TRAIN_OUT}/panel_reference --out-dir $(dirname {output})"
