"""Fixed settings of the CDHscan model. Changing any of them makes the released model invalid."""

BIN_BP = 10_000                  # Hi-C map and WGS track resolution
WINDOW_BINS = 100                # one model window = 1 Mb = 100 bins
CHROMS = [f"chr{i}" for i in range(1, 23)]
ACROCENTRIC_CHROMS = ["chr13", "chr14", "chr15", "chr21", "chr22"]
# The CDH labels (Ni et al. 2026, projected to CHM13) exist only on the non-acrocentric autosomes.
LABELLED_CHROMS = [chrom for chrom in CHROMS if chrom not in ACROCENTRIC_CHROMS]

# Hi-C map channels: own values, own minus the clean panel median, clean panel MAD.
OWN_MAP_VALUES = ["balanced_log_oe", "raw_log_oe", "row_coverage", "column_coverage"]
MAP_CHANNELS = (OWN_MAP_VALUES + [f"{name}_minus_clean_median" for name in OWN_MAP_VALUES]
                + [f"{name}_clean_mad" for name in OWN_MAP_VALUES])

# Read counts per 10 kb bin, in stored column order. The last three are kept for reference only.
COUNT_COLUMNS = ["depth_reads", "all_reads", "soft_clipped", "mate_unmapped", "discordant",
                 "soft_clipped_q20", "mate_unmapped_q20", "discordant_q20"]
MIN_MAPQ = 20                    # depth_reads: reads with at least this mapping quality
MIN_CLIP_BP = 20                 # soft_clipped: at least this many soft-clipped bases at one end

# WGS track channels, in the layout of the Hi-C channels.
WGS_VALUES = ["log_relative_depth", "soft_clipped_fraction", "mate_unmapped_fraction", "discordant_fraction"]
WGS_CHANNELS = (WGS_VALUES + [f"{name}_minus_clean_median" for name in WGS_VALUES]
                + [f"{name}_clean_mad" for name in WGS_VALUES])

# Locus catalogue and windows.
MAX_LOCUS_BP = 1_000_000         # loci are merged CDH regions of at most 1 Mb
MIN_ALIGNED_FRACTION = 0.5       # share of a CHM13 interval that must align to GRCh38 (WGS is on GRCh38)
MIN_CARRIERS = 3                 # training uses only loci with at least this many carriers
MAX_BLANK_FRACTION = 0.2         # windows with more bins lacking a SCALE factor are not scored
REPLACEMENT_FLANK_BP = 50_000    # cosigt picks proxy haplotypes for the locus +-50 kb

# A locus is called a CDH when the mean probability of the seeds exceeds this value.
CDH_PROBABILITY = 0.5
SEEDS = [2026, 2027, 2028]

# Reconstruction on proxy haplotypes (Ni et al. Hi-C settings).
PROXY_SLOTS = ["proxy1", "proxy2"]           # the two individualized references of a person
MAP_ARMS = ["chm13"] + PROXY_SLOTS           # references whose window maps are reported
ALIGNMENT_SLACK_BP = 20_000                  # a proxy allele is aligned to its CHM13 region +-20 kb
VALID_PAIR_TYPES = ("UU", "UR", "RU")        # pairtools pair types with both ends unique
UNIQUE_SIDE_TYPES = ("U", "R")
SPECIFIC_BIN_BP = 25_000                     # profile resolution of haplotype-specific contacts
