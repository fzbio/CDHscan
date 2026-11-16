"""Extract the allele of every GRCh38 region from one HPRC haplotype.

The HPRC BAMs hold whole-contig alignments of one haplotype assembly (query) to GRCh38, split into
primary and supplementary records at large SVs. A reference position is converted to a contig
position by walking the record's CIGAR; hard clips count, so contig positions refer to the full
contig. For reverse-strand records the contig position is mirrored.

Both region ends must be covered by records of one contig and one orientation; the allele is the
contig piece between them, at most MAX_LENGTH_RATIO times the region length (a longer piece would
join two copies of a duplicated sequence). When several record pairs qualify, the one with the
highest summed mapping quality wins (then the one closest to the region length). The BAM and the
bgzipped assembly can be local paths or URLs; their indexes must be local.

Outputs: a table (region, status, contig, start, end, strand, length) and a FASTA with one record
per projected region, named "<region>|<contig>:<start>-<end>" (0-based start).
"""

import numpy as np
import pysam

MAX_LENGTH_RATIO = 3.0

REFERENCE_OPS = {0, 2, 3, 7, 8}          # M, D, N, =, X consume the reference
QUERY_OPS = {0, 1, 4, 5, 7, 8}           # M, I, S, H, =, X consume the query (hard clips included)
ALIGNED_OPS = {0, 7, 8}


class CigarIndex:
    """Reference-to-query position lookup for one alignment record (query positions in alignment orientation)."""

    def __init__(self, record):
        ref_starts, query_starts, lengths, aligned = [], [], [], []
        ref, query = record.reference_start, 0
        for op, length in record.cigartuples:
            if op in REFERENCE_OPS:
                ref_starts.append(ref)
                query_starts.append(query)
                lengths.append(length)
                aligned.append(op in ALIGNED_OPS)
                ref += length
            if op in QUERY_OPS:
                query += length
        self.ref_starts = np.array(ref_starts)
        self.query_starts = np.array(query_starts)
        self.lengths = np.array(lengths)
        self.aligned = np.array(aligned)
        self.contig = record.query_name
        self.contig_length = record.infer_read_length()
        self.reverse = record.is_reverse
        self.mapq = record.mapping_quality
        self.reference_start, self.reference_end = record.reference_start, record.reference_end

    def covers(self, position):
        return self.reference_start <= position < self.reference_end

    def contig_position(self, position):
        """Forward-strand contig position of a reference position; a deleted base maps to the next aligned base."""
        op = np.searchsorted(self.ref_starts, position, side="right") - 1
        offset = position - self.ref_starts[op] if self.aligned[op] else 0
        query = int(self.query_starts[op] + offset)
        return self.contig_length - 1 - query if self.reverse else query


def records_at(bam, chrom, start, end, cache):
    """Primary and supplementary records overlapping [start, end), as cached CigarIndex objects."""
    indexes = []
    for record in bam.fetch(chrom, start, end):
        if record.is_unmapped or record.is_secondary:
            continue
        key = (record.query_name, record.reference_start, record.flag)
        if key not in cache:
            cache[key] = CigarIndex(record)
        indexes.append(cache[key])
    return indexes


def strict_projection(bam, chrom, start, end, cache):
    """(status, contig, low, high, strand) of the strict projection of [start, end)."""
    left = [index for index in records_at(bam, chrom, start, start + 1, cache) if index.covers(start)]
    right = [index for index in records_at(bam, chrom, end - 1, end, cache) if index.covers(end - 1)]
    if not left or not right:
        return ("boundary_unaligned", None, None, None, None)
    region_length = end - start
    best = None
    for left_index in left:
        for right_index in right:
            if left_index.contig != right_index.contig or left_index.reverse != right_index.reverse:
                continue
            first, last = left_index.contig_position(start), right_index.contig_position(end - 1)
            low, high = (last, first + 1) if left_index.reverse else (first, last + 1)
            if high <= low or high - low > MAX_LENGTH_RATIO * region_length:
                continue
            rank = (left_index.mapq + right_index.mapq, -abs((high - low) - region_length))
            if best is None or rank > best[0]:
                best = (rank, left_index.contig, low, high, "-" if left_index.reverse else "+")
    if best is None:
        return ("no_consistent_pair", None, None, None, None)
    return ("ok",) + best[1:]


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def cache_contig_length(cache, contig):
    """Full contig length, from any cached record of the contig."""
    return next(index.contig_length for index in cache.values() if index.contig == contig)


def read_regions(path):
    regions = []
    with open(path) as handle:
        for line in handle:
            chrom, start, end, name = line.split()[:4]
            regions.append((chrom, int(start), int(end), name))
    return regions


def project_haplotype(bam_url, bai, assembly_url, fai, gzi, regions_file, out_table, out_fasta):
    """Project every region of regions_file (BED: chrom, start, end, name) onto one haplotype."""
    bam = pysam.AlignmentFile(bam_url, index_filename=bai)
    assembly = pysam.FastaFile(assembly_url, filepath_index=fai, filepath_index_compressed=gzi)
    cache, n_ok = {}, 0
    regions = read_regions(regions_file)
    with open(out_table, "w") as table, open(out_fasta, "w") as fasta:
        table.write("region\tstatus\tcontig\tstart\tend\tstrand\tlength\n")
        for chrom, start, end, name in regions:
            status, contig, low, high, strand = strict_projection(bam, chrom, start, end, cache)
            if status != "ok":
                table.write(f"{name}\t{status}\t.\t.\t.\t.\t.\n")
                continue
            assert cache_contig_length(cache, contig) == assembly.get_reference_length(contig), \
                f"{contig}: alignment and assembly lengths differ"
            sequence = assembly.fetch(contig, low, high)
            if strand == "-":
                sequence = reverse_complement(sequence)
            table.write(f"{name}\t{status}\t{contig}\t{low}\t{high}\t{strand}\t{high - low}\n")
            fasta.write(f">{name}|{contig}:{low}-{high}\n{sequence}\n")
            n_ok += 1
    print(f"{n_ok} of {len(regions)} regions projected", flush=True)
