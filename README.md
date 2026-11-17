# CDHscan

CDHscan predicts **contact difference hotspots (CDHs)** — loci where the Hi-C contacts of a
person's two haplotypes differ — without assembling the person's genome. Prediction needs only a Hi-C
map aligned to T2T-CHM13 and short-read WGS aligned to GRCh38. At each predicted CDH, it uses
[cosigt](https://github.com/davidebolo1993/cosigt) to pick the two HPRC haplotypes that best
match the person's WGS reads. These two "proxy haplotypes" replace the reference sequence at
the locus, giving two individualized references. The person's own Hi-C reads are then remapped
to each reference to reconstruct the local contact maps.

```
CHM13 Hi-C (.hic) ──┐
                    ├─► CDHscan model ─► CDH probability ─► cosigt ─► proxy haplotypes
GRCh38 WGS (CRAM) ──┘   (Hi-C + WGS)     (12,222 loci)                       │
                                                                             ▼
Hi-C reads (FASTQ) ─────────────────────────► remapped to the two proxy references
                                                    ─► local contact maps (CHM13, proxy 1, proxy 2)
```

The model was trained on 171 HPRC people. Their CDH labels come from Ni et al. (2026), who
compared each person's Hap1 and Hap2 Hi-C maps on the person's own assembly.

## Installation

```bash
git clone https://github.com/fzbio/CDHscan.git && cd CDHscan
mamba env create -f environment.yml      # creates the environment "cdhscan" and installs the package
mamba activate cdhscan
bash scripts/download_model.sh           # released model from Zenodo (0.8 GB)
```

To use a container instead, build the image and run the workflow inside it:

```bash
docker build -t cdhscan:1.0.0 .
apptainer build cdhscan.sif docker-daemon://cdhscan:1.0.0   # for HPC systems without Docker
```

The container holds everything for the `predict`, `reconstruct` and `train` targets. Run the
`cosigt` target from the conda environment: cosigt starts its own containers (Apptainer) or conda
environments.

## Test

```bash
bash scripts/download_test_data.sh   # test data from Zenodo (~115 MB); run_test.sh also does this when needed
bash test/run_test.sh                # predict target, compared with the expected values (a few minutes)
bash test/run_test.sh cosigt         # also the cosigt target (pulls the cosigt containers on first use)
bash test/run_test.sh reconstruct    # cosigt, then the remapping of Hi-C reads to the proxy references
```

The test data (Zenodo, ~115 MB, unpacked into `test/data`) is one person outside the training
set (HG00097). It covers a 1.6 Mb region of CHM13 chr20 with 9 catalogue loci, and includes the
person's Omni-C read pairs near one of them. The data are cut to this region, so the test only
checks that the workflow runs and gives the expected numbers. The test scores do not represent
real performance.

## Usage

Make a sample sheet (tab-separated) and set the paths in `config/config.yaml`:

| column | content |
|---|---|
| `sample` | person name |
| `hic` | Hi-C map on T2T-CHM13 v2.0, `.hic` with 10 kb raw counts and SCALE normalisation (local path or http(s)/s3 URL) |
| `alignment` | WGS reads on GRCh38 (CRAM or BAM, indexed); set `grch38_fasta` to the FASTA used for the CRAMs |

For the `reconstruct` target, also list the Hi-C reads (`reconstruct.hic_reads`, tab-separated:
`sample`, `library`, `r1`, `r2`; FASTQ, local path or http(s)/s3 URL) and set
`reconstruct.chm13_fasta`.

```bash
snakemake predict     --cores 16                   # CDH probabilities
snakemake cosigt      --cores 16                   # proxy haplotypes at the predicted CDHs
snakemake reconstruct --cores 16                   # Hi-C reads remapped to the two proxy references
snakemake predict --executor slurm --jobs 50       # on a cluster (any Snakemake executor plugin)
```

### Outputs

`results/<sample>/predictions.tsv`, one row per catalogue locus:

| column | content |
|---|---|
| `locus_id`, `chrom`, `start`, `end` | catalogue locus (CHM13) |
| `n_carriers` | number of training people (of 171) who carry a CDH here |
| `window_start`, `window_end` | the 1 Mb window the model reads |
| `n_blank_bins`, `status` | bins without a SCALE factor; windows with more than 20 are not scored |
| `probability_seed<k>`, `probability` | CDH probability of each of the 3 trained networks, and their mean |
| `cdh` | `probability` > 0.5 |

`results/<sample>/proxy_haplotypes.tsv` (cosigt target) gives, for each CDH locus of the
person, the GRCh38 region (locus ±50 kb), the two proxy haplotypes and cosigt's cosine
similarity. `proxy_haplotypes.fa` holds their sequences.

`results/<sample>/reconstruction/` (reconstruct target), per reconstructed locus:

| file | content |
|---|---|
| `maps.npz` | `maps`: contact counts on CHM13, proxy 1 and proxy 2, on the 10 kb CHM13 bins of the 1 Mb window (loci × 3 × 100 × 100); `specific_maps`: haplotype-specific contacts (read pair found on one proxy reference only); `specific_maps_own`: the same in each proxy reference's own coordinates; `specific_total`, `specific_count`: 25 kb profiles |
| `contacts.tsv` | contacts and contacts touching the replacement region, per reference |
| `specific.tsv` | contacts and haplotype-specific contacts, over the window and over the locus |
| `region.tsv` | read pairs touching the replacement region on CHM13 and on the proxy pair, and those found on only one of them |
| `<reference>.dedup.pairs.gz` | the non-duplicate window contacts of each reference (pairtools format); duplicate counts per library in `<reference>.<library>.dedup_stats.txt` |

### Input scope

- The model was trained on HPRC Omni-C maps (`chm13.hic`) and 1000 Genomes 30x WGS. Maps from
  other protocols or depths, and WGS from other pipelines, shift the input values.
- Loci are on the non-acrocentric autosomes (the labels exclude chr13, 14, 15, 21, 22, X and Y).
- `cosigt` reads the HPRC assemblies and alignments directly from public S3. The people in the
  sample sheet, and any samples in `cosigt.exclude_panel_samples`, are never used as panel
  haplotypes.
- If bwa-mem2 inside the cosigt container aborts with a CPU-instruction error, set
  `cosigt.bwa_mem2_binary` to a fixed build (for example the AVX2 one; see `config/config.yaml`).

## Method in brief

For each catalogue locus, the model reads a 1 Mb window around the locus.

- **Hi-C input (12 × 100 × 100):** the person's balanced and raw log2(observed/expected) maps
  and the coverage of each bin; each value minus the median of the *clean panel*; and the
  median absolute deviation (MAD) of the clean panel. The clean panel is the training people who
  do not carry a CDH at the locus.
- **WGS input (12 × 100):** read depth and the fractions of soft-clipped, mate-unmapped and
  discordant reads, in the same three parts.

A 2D and a 1D residual CNN read the two inputs. Their outputs are joined, and a small head gives
the CDH probability. During training, each batch holds pairs of one carrier and one non-carrier
at the same locus. Locus frequency is therefore not a learnable shortcut. The
`resources/model` folder ships the clean-panel medians and MADs of every locus (the "panel
reference"), so scoring a new person needs no other person's data.

For reconstruction, each proxy allele is aligned to its CHM13 region (minimap2) and spliced in,
giving two references per person. The alignment also converts every position back to CHM13. All
Hi-C read pairs are first mapped to CHM13 (bwa mem `-5SP -T0`; pairtools, MAPQ ≥ 40, as in Ni et
al.). A pair with an end mapped uniquely outside every locus window cannot form a window contact
on any reference, because outside the swapped regions the references are CHM13. Only the other
pairs are mapped again, to each proxy reference separately. The proxy haplotypes thus supply the
sequence, and the contacts come from the person's own reads.

## Retraining

```bash
snakemake train --configfile config/train.yaml --cores 32
```

This target rebuilds the released model from public data. It downloads the 171 HPRC
`chm13.hic` maps and reads the 1000 Genomes CRAMs (`train.alignment_pattern`). It also uses the
CHM13 CDH labels in `resources/labels` and the training and validation roles in
`resources/training_people.tsv`. The output folder `results/train/model` has the same layout as
the released model. Training needs about 130 GB of memory per seed.

## Citation

Wang F. *et al.* Pangenome-guided recovery of haplotype-differential chromatin contacts in
unassembled genomes (2026).

Released model and test data: Zenodo, doi:[10.5281/zenodo.24273502](https://doi.org/10.5281/zenodo.24273502).
Data of the study (labels, cross-validation models and scores, proxy haplotypes, population scan):
Zenodo, doi:[10.5281/zenodo.24296622](https://doi.org/10.5281/zenodo.24296622).

Data sources: CDH labels from Ni et al. (2026), bioRxiv doi:10.64898/2026.10.05.756789 (Table S5,
projected to CHM13); Hi-C maps and assemblies from the Human Pangenome Reference Consortium;
WGS from the 1000 Genomes Project 30x release; proxy haplotypes from cosigt (Bolognini et al.).

## License

MIT
