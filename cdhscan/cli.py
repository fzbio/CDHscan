"""Command line of CDHscan: one subcommand per workflow step (the Snakemake rules call these)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELLED_CHROMS, SEEDS


def key_value_pairs(items):
    """["HG00097=path", ...] -> {"HG00097": "path", ...}"""
    return dict(item.split("=", 1) for item in items)


# ---- prediction ----

def run_extract_hic(args):
    from .hic import extract_bands
    extract_bands(args.hic, args.chroms, args.out_dir)


def run_count_reads(args):
    from .reads import count_reads, load_bins
    bins = load_bins(args.bins, args.chroms, args.region)
    np.save(args.out, count_reads(args.alignment, args.reference, bins))


def run_score(args):
    import torch
    from .score import score_person
    torch.set_num_threads(args.threads)
    table = score_person(args.model_dir, args.bands, args.counts, args.chroms, args.region)
    table.to_csv(args.out, sep="\t", index=False, float_format="%.4f")


# ---- training ----

def people_table(path):
    return pd.read_csv(path, sep="\t")


def run_bins(args):
    from .training.lift import bin_table
    n_bins = {chrom: len(np.load(Path(args.band_dir) / f"{chrom}.npz")["scale_factor"]) for chrom in args.chroms}
    bin_table(args.chain, n_bins).to_csv(args.out, sep="\t", index=False, float_format="%.4f")


def run_windows(args):
    from .training.catalogue import catalogue, training_windows
    from .training.lift import read_chains
    people = people_table(args.people)
    valid_bins = {person: {chrom: np.isfinite(np.load(Path(args.band_root) / person / f"{chrom}.npz")["scale_factor"])
                           for chrom in LABELLED_CHROMS} for person in people["sample"]}
    n_bins = {chrom: len(valid_bins[people["sample"].iloc[0]][chrom]) for chrom in LABELLED_CHROMS}
    loci = catalogue(args.loci_bed, read_chains(args.chain, LABELLED_CHROMS), n_bins)
    regions = pd.read_csv(args.labels, sep="\t").rename(columns={"chr": "chrom"})
    regions = regions[regions["chrom"].isin(LABELLED_CHROMS) & regions["sample"].isin(people["sample"])]
    windows = training_windows(loci, regions, people, valid_bins, args.seed)
    out_dir = Path(args.out_dir)
    loci.to_csv(out_dir / "loci.tsv", sep="\t", index=False, float_format="%.4f")
    windows.to_csv(out_dir / "windows.tsv", sep="\t", index=False)
    print(f"{len(loci)} loci ({int(loci['scored'].sum())} scored); training windows:\n"
          + windows.groupby(["role", "label"]).size().unstack().to_string(), flush=True)


def value_bands_of(people, band_root, chrom):
    from .features import hic_value_bands
    from .hic import load_bands
    return {person: hic_value_bands(*load_bands(Path(band_root) / person, chrom)) for person in people}


def wgs_values_of(people, counts_dir, bins):
    from .features import wgs_values
    return {person: wgs_values(np.load(Path(counts_dir) / f"{person}.npy"), bins) for person in people}


def first_rows_of(bins):
    return bins.reset_index().groupby("chrom")["index"].min().to_dict()


def run_hic_inputs(args):
    from .training.inputs import carriers_of, write_hic_inputs
    panel = people_table(args.people)["sample"].tolist()
    out_file = Path(args.out)
    windows = pd.read_csv(Path(args.windows_dir) / "windows.tsv", sep="\t")
    windows = windows[windows["input_file"] == out_file.name].sort_values("input_row").reset_index(drop=True)
    assert (windows["input_row"] == np.arange(len(windows))).all(), f"{out_file.name}: rows are not 0..n-1"
    loci = pd.read_csv(Path(args.windows_dir) / "loci.tsv", sep="\t")
    write_hic_inputs(windows, carriers_of(loci), panel, value_bands_of(panel, args.band_root, args.chrom), out_file)


def run_wgs_tracks(args):
    from .reads import load_bins
    from .training.inputs import carriers_of, wgs_tracks
    panel = people_table(args.people)["sample"].tolist()
    windows = pd.read_csv(Path(args.windows_dir) / "windows.tsv", sep="\t")
    loci = pd.read_csv(Path(args.windows_dir) / "loci.tsv", sep="\t")
    bins = load_bins(args.bins, LABELLED_CHROMS)
    tracks = wgs_tracks(windows, carriers_of(loci), panel, wgs_values_of(panel, args.counts_dir, bins), first_rows_of(bins))
    np.save(args.out, tracks)


def run_train(args):
    from .training.train import train_model
    train_model(args.windows, args.hic_inputs_dir, args.tracks, args.seed, args.threads, args.out_dir)


def run_panel_reference(args):
    from .reads import load_bins
    from .training.release import panel_reference
    panel = people_table(args.people)["sample"].tolist()
    loci = pd.read_csv(args.loci, sep="\t")
    loci = loci[loci["scored"] & (loci["chrom"] == args.chrom)]
    bins = load_bins(args.bins, LABELLED_CHROMS)
    panel_reference(loci, panel, value_bands_of(panel, args.band_root, args.chrom),
                    wgs_values_of(panel, args.counts_dir, bins), first_rows_of(bins)[args.chrom], args.out)


def run_package(args):
    from .training.release import package_model
    package_model(args.loci, args.bins, args.seed_dirs, args.panel_reference_dir, LABELLED_CHROMS, args.out_dir)


# ---- cosigt ----

def run_cosigt_regions(args):
    from .cosigt.regions import select_regions
    select_regions(args.catalogue, key_value_pairs(args.predictions), args.out_dir, args.extra_loci,
                   include_calls=not args.extra_only)


def run_project_haplotype(args):
    from .cosigt.project import project_haplotype
    project_haplotype(args.bam, args.bai, args.assembly, args.fai, args.gzi, args.regions, args.out_table, args.out_fasta)


def run_collect_alleles(args):
    from .cosigt.alleles import collect_alleles
    collect_alleles(args.regions, args.fastas, args.grch38, args.out_dir)


def run_configure_cosigt(args):
    from .cosigt.configure import configure_cosigt
    configure_cosigt(key_value_pairs(args.alignments), args.regions, args.alleles_dir, args.reference,
                     args.pipeline_dir, args.results_dir, args.tmp_dir)


def run_summarize_cosigt(args):
    from .cosigt.summarize import summarize_person
    summarize_person(args.sample, args.calls, args.results_dir, args.alleles_dir, args.out_table, args.out_fasta)


# ---- reconstruction ----

def run_reconstruct_plan(args):
    from .reconstruct.plan import plan_person
    plan_person(args.proxies, args.catalogue, args.alleles_dir, args.out_dir)


def run_build_reference(args):
    from .reconstruct.reference import build_reference
    build_reference(args.plan, args.loci, args.slot, args.chm13, args.out_dir)


def run_window_pairs(args):
    from .reconstruct.pairs import filter_pairs
    filter_pairs(args.windows, args.stage2_ids)


def run_reconstruct_maps(args):
    from .reconstruct.maps import window_maps
    window_maps(args.loci, args.pairs_dir, args.references_dir, args.out_dir)


def main():
    parser = argparse.ArgumentParser(prog="cdhscan", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, function, help_text, arguments):
        sub = commands.add_parser(name, help=help_text)
        for flags, options in arguments:
            sub.add_argument(*flags, **options)
        sub.set_defaults(function=function)

    required = {"required": True}
    chroms = (["--chroms"], {"nargs": "+", "default": LABELLED_CHROMS})
    region = (["--region"], {"help": "restrict to one CHM13 region chrom:start-end (tests only)"})

    command("extract-hic", run_extract_hic, "Hi-C bands of one person from a CHM13 .hic file",
            [(["--hic"], required), chroms, (["--out-dir"], required)])
    command("count-reads", run_count_reads, "WGS read counts of one person in the 10 kb bins",
            [(["--alignment"], required), (["--reference"], {"help": "GRCh38 FASTA (needed for CRAM)"}),
             (["--bins"], required), chroms, region, (["--out"], required)])
    command("score", run_score, "CDH probability of every catalogue locus of one person",
            [(["--model-dir"], required), (["--bands"], required), (["--counts"], required), chroms, region,
             (["--threads"], {"type": int, "default": 1}), (["--out"], required)])

    command("bins", run_bins, "10 kb CHM13 bins lifted to GRCh38",
            [(["--chain"], required), (["--band-dir"], required), chroms, (["--out"], required)])
    command("windows", run_windows, "locus catalogue and training windows",
            [(["--loci-bed"], required), (["--labels"], required), (["--people"], required),
             (["--band-root"], required), (["--chain"], required), (["--seed"], {"type": int, "default": SEEDS[0]}),
             (["--out-dir"], required)])
    command("hic-inputs", run_hic_inputs, "Hi-C inputs of one stored part of the training windows",
            [(["--windows-dir"], required), (["--people"], required), (["--band-root"], required),
             (["--chrom"], required), (["--out"], required)])
    command("wgs-tracks", run_wgs_tracks, "WGS inputs of the training windows",
            [(["--windows-dir"], required), (["--people"], required), (["--counts-dir"], required),
             (["--bins"], required), (["--out"], required)])
    command("train", run_train, "train the network with one seed",
            [(["--windows"], required), (["--hic-inputs-dir"], required), (["--tracks"], required),
             (["--seed"], {"type": int, "required": True}), (["--threads"], {"type": int, "default": 8}),
             (["--out-dir"], required)])
    command("panel-reference", run_panel_reference, "clean panel median and MAD of the loci of one chromosome",
            [(["--loci"], required), (["--people"], required), (["--band-root"], required),
             (["--counts-dir"], required), (["--bins"], required), (["--chrom"], required), (["--out"], required)])
    command("package", run_package, "assemble the model folder",
            [(["--loci"], required), (["--bins"], required), (["--seed-dirs"], {"nargs": "+", "required": True}),
             (["--panel-reference-dir"], required), (["--out-dir"], required)])

    command("cosigt-regions", run_cosigt_regions, "GRCh38 regions of the loci called as CDH",
            [(["--catalogue"], required), (["--predictions"], {"nargs": "+", "required": True, "help": "sample=path"}),
             (["--extra-loci"], {"nargs": "*", "default": [], "help": "locus ids genotyped in every person"}),
             (["--extra-only"], {"action": "store_true", "help": "genotype the extra loci only"}),
             (["--out-dir"], required)])
    command("project-haplotype", run_project_haplotype, "alleles of one HPRC haplotype at the regions",
            [(["--bam"], required), (["--bai"], required), (["--assembly"], required), (["--fai"], required),
             (["--gzi"], required), (["--regions"], required), (["--out-table"], required), (["--out-fasta"], required)])
    command("collect-alleles", run_collect_alleles, "one allele FASTA per region for cosigt",
            [(["--regions"], required), (["--fastas"], {"nargs": "*", "default": []}), (["--grch38"], required),
             (["--out-dir"], required)])
    command("configure-cosigt", run_configure_cosigt, "write the cosigt configuration",
            [(["--alignments"], {"nargs": "+", "required": True, "help": "sample=path"}), (["--regions"], required),
             (["--alleles-dir"], required), (["--reference"], required), (["--pipeline-dir"], required),
             (["--results-dir"], required), (["--tmp-dir"], required)])
    command("summarize-cosigt", run_summarize_cosigt, "proxy haplotypes of one person",
            [(["--sample"], required), (["--calls"], required), (["--results-dir"], required),
             (["--alleles-dir"], required), (["--out-table"], required), (["--out-fasta"], required)])

    command("reconstruct-plan", run_reconstruct_plan, "loci and proxy alleles of one person's two references",
            [(["--proxies"], required), (["--catalogue"], required), (["--alleles-dir"], required),
             (["--out-dir"], required)])
    command("build-reference", run_build_reference, "CHM13 with the person's regions swapped for one proxy",
            [(["--plan"], required), (["--loci"], required), (["--slot"], required), (["--chm13"], required),
             (["--out-dir"], required)])
    command("window-pairs", run_window_pairs, "window contacts of a pairtools stream (stdin -> stdout)",
            [(["--windows"], required), (["--stage2-ids"], {"help": "write the read IDs for the second mapping"})])
    command("reconstruct-maps", run_reconstruct_maps, "window maps of one person on CHM13 and the proxy references",
            [(["--loci"], required), (["--pairs-dir"], required), (["--references-dir"], required),
             (["--out-dir"], required)])

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
