#!/usr/bin/env python3
"""
orf_finder.py - ORF prediction in bacterial genomes.

Biological Sequence Analysis homework, programming task.

Pipeline (each stage is a separate section of this file):
    1. Enumeration   - find every open reading frame candidate in all six frames
    2. Length filter - genome-specific minimum length from a random-sequence null model
    3. Start choice  - pick the most likely start codon using the ribosome binding site
    4. Coding score  - self-trained log-odds score (coding model vs background model)
    5. Overlaps      - remove "shadow" ORFs that overlap a better-scoring ORF; write output

Conventions used throughout:
    * Coordinates in output files are 1-based and inclusive (as in GFF), with
      start < end on both strands; the strand column says which way the gene reads.
    * Internally, each strand is handled as its own 5'->3' string. For the minus
      strand this is the reverse complement, and positions are 0-based,
      half-open ("strand-local" coordinates). They are converted to genome
      coordinates only when results are written.
    * Genetic code: NCBI translation table 11 (bacteria, archaea, plastids).

Requires: Python >= 3.9, Biopython.
"""

import argparse
import bisect
import math
import re
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field

from Bio import SeqIO
from Bio.Seq import reverse_complement


# =============================================================================
# Section 1. Enumeration of ORF candidates
# =============================================================================
#
# Definition used here: an ORF candidate is a stretch of codons, in one reading
# frame, that ends with a stop codon and contains no other in-frame stop.
# Every stop codon therefore defines at most one candidate. All in-frame start
# codons between the previous stop and this stop are kept as a list of
# alternative starts; which one is the real start is decided in Section 3.
# Until then the most upstream start (= the longest ORF) is used.

# NCBI table 11 stop codons.
STOP_CODONS = ("TAA", "TAG", "TGA")

# NCBI table 11 start codons used by bacteria. ATG starts ~80-90% of genes,
# GTG ~10% and TTG a few percent. Accepting only ATG would miss real genes.
# Table 11 also lists rarer starts (CTG, ATT, ATC, ATA), but they are used by
# very few genes and would add many false alternative starts, so they are
# left out.
DEFAULT_START_CODONS = ("ATG", "GTG", "TTG")

# A run of at least this many non-ACGT bases (e.g. NNNN... in a scaffold gap)
# is treated as a break in the sequence: ORFs end there and are flagged
# partial. Isolated ambiguous bases are tolerated. A codon containing them
# simply counts as neither start nor stop.
DEFAULT_GAP_RUN = 10


@dataclass
class ORF:
    """One ORF candidate (one stop codon plus its alternative starts).

    All positions are strand-local, 0-based, half-open (see module docstring).
    """
    contig: str
    contig_len: int
    strand: str              # '+' or '-'
    frame: int               # 0, 1 or 2: position of the first codon modulo 3 (strand-local)
    region_start: int        # first base after the upstream in-frame stop (or after a contig edge/gap)
    end: int                 # end of the ORF, including the stop codon (exclusive)
    starts: list             # positions of in-frame start codons, upstream-most first
    partial5: bool           # no upstream in-frame stop seen: the real start may lie beyond the edge
    partial3: bool           # sequence ends (edge or N-gap) before a stop codon is reached
    start: int = field(init=False)   # currently chosen start (updated in Section 3)
    # Filled in by Section 3 (start selection):
    rbs_motif: str = field(default="-", init=False)    # best Shine-Dalgarno match upstream of the chosen start
    rbs_spacer: int = field(default=-1, init=False)    # bp between that motif and the start codon
    start_score: float = field(default=0.0, init=False)
    # Filled in by Section 4 (coding score):
    coding_score: float = field(default=0.0, init=False)   # total log2-odds, coding vs background

    @property
    def coding_end(self) -> int:
        """End of the coding part (the stop codon excluded), strand-local."""
        return self.end if self.partial3 else self.end - 3

    def __post_init__(self):
        # Default start = most upstream one, i.e. the longest possible ORF.
        # For a 5'-partial ORF the longest version starts at the edge itself.
        if self.partial5:
            self.start = self.region_start
        else:
            self.start = self.starts[0]

    @property
    def length(self) -> int:
        """ORF length in nucleotides, from the chosen start to the end of the stop codon."""
        return self.end - self.start

    @property
    def max_codons(self) -> int:
        """Sense codons in the longest version of this ORF (most upstream start, stop not counted)."""
        first = self.region_start if self.partial5 else self.starts[0]
        stop_len = 0 if self.partial3 else 3
        return (self.end - stop_len - first) // 3

    def genome_coords(self):
        """Return (start, end) as 1-based inclusive coordinates on the forward strand."""
        if self.strand == "+":
            return self.start + 1, self.end
        # Minus strand: strand-local [a, b) corresponds to forward [L-b, L-a).
        return self.contig_len - self.end + 1, self.contig_len - self.start


def read_genome(path: str):
    """Yield (contig_id, sequence) for every record in a (multi-)FASTA file.

    Sequences are upper-cased, so soft-masked (lowercase) bases are treated
    like any other base.
    """
    for record in SeqIO.parse(path, "fasta"):
        yield record.id, str(record.seq).upper()


def find_codons(seq: str, codons) -> list:
    """Return every position where one of `codons` occurs, in any frame.

    The lookahead (?=...) makes the regex report overlapping matches, so for
    example both the TAA and the TAG in 'TAATAG' are found. Bases other than
    ACGT never match, so a codon containing N is neither a start nor a stop.
    """
    pattern = re.compile("(?=(" + "|".join(codons) + "))")
    return [m.start() for m in pattern.finditer(seq)]


def ungapped_segments(seq: str, gap_run: int) -> list:
    """Split a sequence into (start, end) pieces separated by runs of >= gap_run non-ACGT bases."""
    segments, pos = [], 0
    for gap in re.finditer("[^ACGT]{%d,}" % gap_run, seq):
        if gap.start() > pos:
            segments.append((pos, gap.start()))
        pos = gap.end()
    if pos < len(seq):
        segments.append((pos, len(seq)))
    return segments


def orfs_on_strand(contig, seq, strand, start_codons, gap_run):
    """Enumerate ORF candidates on one strand (seq is that strand read 5'->3').

    Approach: instead of walking codon by codon, find all stop and start
    codons once with a regex, then sort them into the three frames by
    position modulo 3. Consecutive stops in the same frame bound one
    candidate. This is the same idea OrfM uses (scan the nucleotide
    sequence for stops directly instead of translating six frames), and it
    keeps the code short.
    """
    stops = find_codons(seq, STOP_CODONS)
    starts = find_codons(seq, start_codons)
    orfs = []

    for frame in range(3):
        stops_f = [p for p in stops if p % 3 == frame]
        starts_f = [p for p in starts if p % 3 == frame]

        # Each gap-free piece of sequence is processed on its own.
        for seg_start, seg_end in ungapped_segments(seq, gap_run):
            first_codon = seg_start + (frame - seg_start) % 3
            last_codon_end = first_codon + (seg_end - first_codon) // 3 * 3

            # Stops of this frame that lie completely inside the piece.
            lo = bisect.bisect_left(stops_f, first_codon)
            hi = bisect.bisect_right(stops_f, seg_end - 3)

            # Each stop closes one region: [previous stop end, this stop end).
            # After the last stop, the remaining codons form an open region
            # that reaches the end of the piece without a stop (partial3).
            region_start = first_codon
            regions = []
            for q in stops_f[lo:hi]:
                regions.append((region_start, q + 3, False))
                region_start = q + 3
            regions.append((region_start, last_codon_end, True))

            for r_start, r_end, partial3 in regions:
                # The first region of a piece has no upstream stop, so its
                # real start could lie beyond the edge (partial5).
                partial5 = (r_start == first_codon)
                coding_end = r_end if partial3 else r_end - 3   # exclude the stop codon
                if coding_end - r_start < 3:
                    continue                                     # no room for any codon

                a = bisect.bisect_left(starts_f, r_start)
                b = bisect.bisect_right(starts_f, coding_end - 3)
                region_starts = starts_f[a:b]

                # Keep the candidate if it has a start codon, or if its start
                # may lie beyond the edge (5'-partial). Stop-to-stop stretches
                # without any start codon cannot be genes and are dropped.
                if region_starts or partial5:
                    orfs.append(ORF(contig, len(seq), strand, frame, r_start, r_end,
                                    region_starts, partial5, partial3))
    return orfs


def enumerate_orfs(path, start_codons=DEFAULT_START_CODONS, gap_run=DEFAULT_GAP_RUN):
    """Enumerate ORF candidates on both strands of every contig in a FASTA file.

    Contigs are treated as linear. On a circular chromosome, a gene that
    crosses the origin of the sequence file is reported as two partial ORFs.

    Returns (orfs, genome) where genome maps contig id -> (forward, reverse complement).
    """
    orfs, genome = [], {}
    for contig, seq in read_genome(path):
        rc = reverse_complement(seq)
        genome[contig] = (seq, rc)
        orfs += orfs_on_strand(contig, seq, "+", start_codons, gap_run)
        orfs += orfs_on_strand(contig, rc, "-", start_codons, gap_run)
    return orfs, genome


# =============================================================================
# Section 2. Genome-specific minimum length (null model + FDR)
# =============================================================================
#
# Every DNA sequence contains ORFs, even a random one, so length alone does not
# make a gene. How long random ORFs get depends on base composition: stop
# codons (TAA, TAG, TGA) are AT-rich, so they are rarer in GC-rich genomes and
# random ORFs there are longer. A fixed cutoff such as "100 codons" is
# therefore too lenient for GC-rich genomes and too strict for AT-rich ones.
#
# Null model (generalising Problem 1.14 of Borodovsky & Ekisheva): the genome
# is an i.i.d. sequence with the genome's own base frequencies. Then
#   p = P(codon is a stop)            = P(TAA) + P(TAG) + P(TGA)
#   s = P(codon is a start | not stop) = (P(ATG) + P(GTG) + P(TTG)) / (1 - p)
# and the number of sense codons between two stops is geometric.
#
# Derivation of the expected number of random ORFs with >= L codons:
# take a stop-to-stop region with n sense codons (probability p(1-p)^n).
# Its longest ORF (most upstream start) has >= L codons iff a start codon
# occurs among its first n-L+1 codons. Summing over n >= L gives, per stop,
#   P(longest ORF >= L) = (1-p)^L * s / (p + s - p*s).
# A random genome of size G has about 2*G*p stop codons (G codon positions
# per strand, both strands), so
#   E_null(L) = 2 * G * p * (1-p)^L * s / (p + s - p*s).
#
# Cutoff rule: compare E_null(L) with the number of ORFs actually observed
# with >= L codons, Obs(L). Their ratio estimates the false discovery rate of
# the rule "keep ORFs with >= L codons" (this is the ORF-vs-random plot from
# the lecture, turned into a rule). The cutoff is the smallest L such that
# FDR(L') <= alpha for every L' >= L.
#
# Known limitation: in a real genome, the "noise" ORFs are mostly shadows of
# real genes in other frames, which are longer than i.i.d. random ORFs
# (especially in GC-rich genomes). The FDR is therefore an optimistic
# estimate; Sections 4 and 5 deal with shadow ORFs.

# Default: a lenient pre-filter. The length step only thins out the
# candidates; the coding score (Section 4) makes the real decision. On
# simulated genomes a strict cutoff (0.05) lost ~20% of genes at 66% GC with
# almost no precision gain once the coding score was applied.
DEFAULT_FDR = 0.5


def base_frequencies(genome):
    """Return base frequencies over both strands {A, C, G, T}.

    Counting both strands makes the frequencies strand-symmetric
    (f(A) = f(T), f(C) = f(G)), which is right because ORFs are searched
    on both strands. Non-ACGT bases are ignored.
    """
    counts = {b: 0 for b in "ACGT"}
    for seq, _ in genome.values():
        for b in "ACGT":
            counts[b] += seq.count(b)
    at = (counts["A"] + counts["T"]) / 2
    gc = (counts["C"] + counts["G"]) / 2
    total = 2 * (at + gc)
    return {"A": at / total, "T": at / total, "C": gc / total, "G": gc / total}


def codon_probability(codons, freq):
    """Probability of drawing any of `codons` under the i.i.d. base model."""
    return sum(freq[c[0]] * freq[c[1]] * freq[c[2]] for c in codons)


class NullModel:
    """i.i.d. random-sequence model for ORF lengths (see Section 2 header)."""

    def __init__(self, freq, genome_size, start_codons):
        self.freq = freq
        self.genome_size = genome_size
        self.p = codon_probability(STOP_CODONS, freq)
        self.s = codon_probability(start_codons, freq) / (1 - self.p)

    def tail(self, L):
        """P(the longest ORF ending at a given stop has >= L sense codons)."""
        p, s = self.p, self.s
        return (1 - p) ** L * s / (p + s - p * s)

    def expected(self, L):
        """Expected number of ORFs with >= L codons in a random genome of this size."""
        return 2 * self.genome_size * self.p * self.tail(L)


def observed_tail_counts(orfs, max_len):
    """Obs[L] = number of complete ORF candidates with >= L codons, for L = 0..max_len."""
    hist = [0] * (max_len + 2)
    for o in orfs:
        if not (o.partial5 or o.partial3):
            hist[min(o.max_codons, max_len + 1)] += 1
    obs, running = [0] * (max_len + 2), 0
    for L in range(max_len + 1, -1, -1):
        running += hist[L]
        obs[L] = running
    return obs


def fdr_length_cutoff(orfs, null, alpha, max_len=2000):
    """Smallest L (codons) such that E_null(L') / Obs(L') <= alpha for all L' >= L.

    Returns (cutoff, table) where table holds (L, observed, expected, fdr) rows
    for reporting and plotting.
    """
    obs = observed_tail_counts(orfs, max_len)
    table = []
    for L in range(max_len + 1):
        e = null.expected(L)
        fdr = e / obs[L] if obs[L] else float("inf")
        table.append((L, obs[L], e, fdr))

    # Walk down from the longest observed ORF while the FDR stays <= alpha.
    # (Above the longest ORF nothing is observed, so the FDR is undefined.)
    longest = max(L for L in range(max_len + 1) if obs[L] > 0) if obs[0] else 0
    cutoff = longest + 1                       # = "nothing passes" if even the longest fails
    for L in range(longest, -1, -1):
        if table[L][3] > alpha:
            break
        cutoff = L
    return cutoff, table


def shuffled_genome(genome, seed=1):
    """Return a copy of the genome with each contig's bases randomly shuffled.

    This keeps the exact base composition but destroys every gene, so the
    ORFs found in it are random ORFs. Used to check the analytic null model.
    """
    import random
    rng = random.Random(seed)
    out = {}
    for contig, (seq, _) in genome.items():
        bases = list(seq)
        rng.shuffle(bases)
        shuffled = "".join(bases)
        out[contig] = (shuffled, reverse_complement(shuffled))
    return out


def check_null_with_shuffle(genome, null, start_codons, gap_run, lengths=(30, 60, 100, 150, 200)):
    """Compare E_null(L) with ORF counts in a shuffled copy of the genome.

    Returns rows (L, expected, found_in_shuffle).
    """
    shuffled_orfs = []
    for contig, (seq, rc) in shuffled_genome(genome).items():
        shuffled_orfs += orfs_on_strand(contig, seq, "+", start_codons, gap_run)
        shuffled_orfs += orfs_on_strand(contig, rc, "-", start_codons, gap_run)
    obs = observed_tail_counts(shuffled_orfs, max(lengths))
    return [(L, null.expected(L), obs[L]) for L in lengths]


def write_length_table(table, path):
    """Write the observed-vs-null length table (input for the ORF length plot in the report)."""
    with open(path, "w") as out:
        out.write("codons\tobserved_ge_L\texpected_null_ge_L\tfdr\n")
        for L, o, e, f in table:
            out.write(f"{L}\t{o}\t{e:.4g}\t{f:.4g}\n")


# =============================================================================
# Section 3. Start codon selection (ribosome binding site + start codon type)
# =============================================================================
#
# Most ORFs contain several in-frame start codons, and the most upstream one
# (= the longest ORF) is not always the real start. Under a random model it
# is right only ~75% of the time (Problem 1.15, Borodovsky & Ekisheva), and
# benchmarks show that real gene finders also tend to pick starts too far
# upstream (Korandla et al. 2020, AssessORF). Two signals help:
#
#   1. Ribosome binding site (RBS). In bacteria, translation starts where the
#      3' end of 16S rRNA pairs with a purine-rich Shine-Dalgarno (SD)
#      sequence (consensus AGGAGG) located ~5-10 bp upstream of the start.
#   2. Start codon type. ATG is used far more often than GTG or TTG.
#
# Every alternative start gets   score = RBS score + start codon log-odds,
# and the highest-scoring start wins (ties go to the upstream start).
#
# The RBS score uses a fixed consensus: the longest exact piece of AGGAGG
# found upstream, weighted by how close its spacer is to the optimum. The
# weights are heuristic, on a scale comparable to log2 odds (a full 6-mer
# at optimal spacing = 4, i.e. about as informative as ATG vs TTG). The RBS
# is used as a bonus only, never as a requirement: leaderless genes have no
# SD at all (Wu Orr et al. 2020).
#
# The start codon log-odds are self-trained on the genome (see
# train_start_codon_odds) because start codon usage differs between species.

SD_CONSENSUS = "AGGAGG"

# All exact sub-motifs of AGGAGG with at least 3 bases, with their strength.
SD_MOTIFS = sorted({SD_CONSENSUS[i:j]: j - i - 2          # 3-mer = 1 ... 6-mer = 4
                    for i in range(len(SD_CONSENSUS))
                    for j in range(i + 3, len(SD_CONSENSUS) + 1)}.items(),
                   key=lambda kv: -kv[1])

# Spacer = bases between the end of the SD motif and the first base of the start codon.
SPACER_MIN, SPACER_MAX = 3, 15


def spacer_weight(spacer):
    """Weight for an SD motif at this distance from the start (optimum 5-10 bp)."""
    if 5 <= spacer <= 10:
        return 1.0
    if 3 <= spacer <= 12:
        return 0.75
    return 0.5                                        # 13-15 bp


def rbs_score(seq, start):
    """Best Shine-Dalgarno match upstream of a start codon.

    Returns (score, motif, spacer); score 0 and motif '-' if nothing is found.
    """
    window_start = max(0, start - SPACER_MAX - len(SD_CONSENSUS))
    window = seq[window_start:start - SPACER_MIN]
    best = (0.0, "-", -1)
    for motif, strength in SD_MOTIFS:
        pos = window.find(motif)
        while pos != -1:
            spacer = start - (window_start + pos + len(motif))
            score = strength * spacer_weight(spacer)
            if score > best[0]:
                best = (score, motif, spacer)
            pos = window.find(motif, pos + 1)
    return best


# A real start codon must score at least this much to replace the edge
# start of a 5'-partial ORF (otherwise the gene is assumed to continue
# beyond the contig edge).
EDGE_MIN_SCORE = 3.0

# Training set for start codon usage: starts with an SD motif at least this
# strong are taken as reliable.
STRONG_RBS = 3.0


def train_start_codon_odds(orfs, genome, freq, start_codons):
    """Self-train log2-odds of each start codon type: log2 P(type | real start) / P(type | random).

    Training set: complete ORFs that passed the length cutoff, taking in each
    one the most upstream start that has a strong RBS (an SD 5-mer or better
    at a good spacing). That avoids circularity: the training starts are
    picked by the RBS alone, without using the codon type.
    The background is how often each start codon occurs by chance given the
    genome's base composition. Pseudocount 1 per type.
    """
    counts = {c: 1 for c in start_codons}
    for o in orfs:
        if o.partial5 or o.partial3:
            continue
        seq = genome[o.contig][0 if o.strand == "+" else 1]
        for s in o.starts:
            if rbs_score(seq, s)[0] >= STRONG_RBS:
                counts[seq[s:s + 3]] += 1
                break
    total = sum(counts.values())
    background = {c: codon_probability([c], freq) for c in start_codons}
    bg_total = sum(background.values())
    odds = {c: math.log2((counts[c] / total) / (background[c] / bg_total)) for c in start_codons}
    return odds, counts


def choose_starts(orfs, genome, start_odds, min_codons):
    """Pick the best-scoring start for every ORF (modifies the ORFs in place).

    Only starts that keep the ORF at or above the length cutoff are
    considered, so start selection cannot turn a long ORF into a short one.
    """
    for o in orfs:
        seq = genome[o.contig][0 if o.strand == "+" else 1]
        stop_len = 0 if o.partial3 else 3
        best = None
        for s in o.starts:                                  # upstream first, so ties keep upstream
            if (o.end - stop_len - s) // 3 < min_codons:
                break                                       # further starts are even shorter
            r_score, motif, spacer = rbs_score(seq, s)
            score = r_score + start_odds[seq[s:s + 3]]
            if best is None or score > best[0]:
                best = (score, s, motif, spacer)

        if best is None:
            continue                                        # keep the default start
        if o.partial5 and best[0] < EDGE_MIN_SCORE:
            continue                                        # keep the edge start
        o.start_score, o.start, o.rbs_motif, o.rbs_spacer = best


# =============================================================================
# Section 4. Coding score: three-periodic Markov chain vs background
# =============================================================================
#
# Length alone cannot tell a real gene from a "shadow" ORF: the same stretch of
# DNA read in another frame, or on the other strand. Shadows are common
# because a real gene suppresses stop codons in its own frame, and (in
# GC-rich genomes) the other frames are stop-poor too. What differs is the
# periodic composition of coding DNA: each codon position (1st, 2nd, 3rd)
# has its own base statistics, and those statistics only line up in the
# real frame.
#
# Coding model: a three-periodic (inhomogeneous) Markov chain of order k
# (Problem 3.12 of Borodovsky & Ekisheva, generalised to order k):
#   P(x_i | x_{i-k} .. x_{i-1}, codon position of x_i)
# With k = 5, each base is predicted from the previous five (a hexamer
# model), the same order GENSCAN uses for coding regions, and Prodigal also
# scores genes by hexamer statistics.
#
# Background model: an ordinary (homogeneous) Markov chain of the same order
# trained on the whole genome, both strands. It knows the genome's
# composition but not the reading frame.
#
# Score of an ORF = sum over its bases of log2( P_coding / P_background ),
# i.e. the log-odds discrimination from the lecture's CpG island example.
# Score > 0 means the ORF is more likely under the coding model.
#
# Self-training (GeneMarkS-style): no annotation is used.
#   round 1: train on long ORFs that do not overlap each other (taking the
#            longest first, so most shadows are excluded)
#   round 2+: re-train on the ORFs that scored > 0 in the previous round
#             (again without overlaps, taking the best-scoring first)

DEFAULT_ORDER = 5
DEFAULT_ROUNDS = 3
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

# Maximum overlap (bp) between two accepted ORFs, on either strand. Used both
# for building training sets here and for the final gene set (Section 5).
MAX_OVERLAP = 60


def kmer_indices(seq, k):
    """For each position i, the index of the (k+1)-mer ending at i (context + base).

    Bases are coded 2 bits each, so a (k+1)-mer is an integer in [0, 4^(k+1)).
    Positions whose window contains a non-ACGT base (or lies before position
    k) get the special index 4^(k+1), which always scores 0.
    """
    width = k + 1
    mask = 4 ** width - 1
    invalid = 4 ** width
    out = array("i", [invalid]) * len(seq)
    value, run = 0, 0
    for i, base in enumerate(seq):
        code = BASE_CODE.get(base)
        if code is None:
            value, run = 0, 0
            continue
        value = ((value << 2) | code) & mask
        run += 1
        if run >= width:
            out[i] = value
    return out


def conditional_log2(counts, k, pseudocount=1.0):
    """Turn (k+1)-mer counts into log2 P(base | k-mer context), with pseudocounts.

    `counts` maps (k+1)-mer index -> count. The four (k+1)-mers that share a
    context are consecutive indices (context * 4 + base).
    """
    n = 4 ** (k + 1)
    table = [0.0] * n
    for ctx in range(4 ** k):
        cells = [counts.get(ctx * 4 + b, 0) + pseudocount for b in range(4)]
        total = sum(cells)
        for b in range(4):
            table[ctx * 4 + b] = math.log2(cells[b] / total)
    return table


class CodingModel:
    """Three-periodic order-k coding model and order-k background, as one log-odds table per codon position."""

    def __init__(self, k, background_counts):
        self.k = k
        self.background = conditional_log2(background_counts, k)
        self.log_odds = None                  # filled by train()

    def train(self, orfs, indices):
        """Count (k+1)-mers at each codon position in the given ORFs and build the log-odds tables."""
        counts = [Counter(), Counter(), Counter()]
        for o in orfs:
            idx = indices[o.contig][0 if o.strand == "+" else 1]
            for phase in range(3):
                counts[phase].update(idx[self._first(o, phase):o.coding_end:3])
        invalid = 4 ** (self.k + 1)
        self.log_odds = []
        for phase in range(3):
            counts[phase].pop(invalid, None)
            coding = conditional_log2(counts[phase], self.k)
            table = [c - b for c, b in zip(coding, self.background)]
            table.append(0.0)                 # the 'invalid' index scores 0
            self.log_odds.append(table)

    def _first(self, o, phase):
        """First position >= start + k whose codon position is `phase` (0 = first base of a codon)."""
        first = o.start + self.k
        return first + (phase - (first - o.start)) % 3

    def score(self, o, indices):
        """Total log2-odds of an ORF (from its chosen start to the base before the stop)."""
        idx = indices[o.contig][0 if o.strand == "+" else 1]
        return sum(sum(map(self.log_odds[phase].__getitem__,
                           idx[self._first(o, phase):o.coding_end:3]))
                   for phase in range(3))


def select_non_overlapping(orfs, key, max_overlap):
    """Greedy selection: take ORFs best-first by `key`, skip any that overlaps an
    already accepted ORF (either strand) by more than `max_overlap` bp.
    """
    accepted = {}                                    # contig -> sorted list of (start, end, orf)
    chosen = []
    for o in sorted(orfs, key=key, reverse=True):
        a, b = o.genome_coords()
        lst = accepted.setdefault(o.contig, [])
        i = bisect.bisect_left(lst, (a,))
        clash = False
        j = i - 1                                    # accepted ORFs starting before this one
        while j >= 0 and lst[j][1] >= a:
            if min(b, lst[j][1]) - a + 1 > max_overlap:
                clash = True
                break
            j -= 1
        j = i                                        # accepted ORFs starting inside this one
        while not clash and j < len(lst) and lst[j][0] <= b:
            if min(b, lst[j][1]) - lst[j][0] + 1 > max_overlap:
                clash = True
            j += 1
        if not clash:
            lst.insert(i, (a, b, o))
            chosen.append(o)
    return chosen


def build_indices(genome, k):
    """(k+1)-mer index arrays for both strands of every contig, plus genome-wide background counts."""
    indices, background = {}, Counter()
    for contig, (seq, rc) in genome.items():
        fwd, rev = kmer_indices(seq, k), kmer_indices(rc, k)
        indices[contig] = (fwd, rev)
        background.update(fwd)
        background.update(rev)
    background.pop(4 ** (k + 1), None)
    return indices, background


def self_train_and_score(orfs, genome, k, rounds, log):
    """Self-train the coding model (see Section 4 header) and set o.coding_score for every ORF."""
    indices, background = build_indices(genome, k)
    model = CodingModel(k, background)

    complete = [o for o in orfs if not (o.partial5 or o.partial3)]
    training = select_non_overlapping(complete, key=lambda o: o.length,
                                      max_overlap=MAX_OVERLAP)
    for r in range(1, rounds + 1):
        model.train(training, indices)
        for o in orfs:
            o.coding_score = model.score(o, indices)
        positive = [o for o in orfs if o.coding_score > 0]
        log(f"    round {r}: trained on {len(training)} ORFs -> {len(positive)} ORFs score > 0")
        training = select_non_overlapping([o for o in positive if not (o.partial5 or o.partial3)],
                                          key=lambda o: o.coding_score,
                                          max_overlap=MAX_OVERLAP)
    return model


# =============================================================================
# Section 5. Overlap resolution and output
# =============================================================================
#
# After Section 4, a real gene and a shadow ORF on the opposite strand can both
# score > 0, because the shadow reads the same coding DNA backwards. Real
# bacterial genes rarely overlap by more than a few dozen bp (typical
# overlaps are 1-4 bp in operons, e.g. ...ATGA where one gene's stop and the
# next gene's start share bases). So the final gene set is built greedily:
# ORFs are ranked by their total evidence (coding log-odds + start score),
# and an ORF is accepted only if it overlaps every already-accepted ORF by
# at most MAX_OVERLAP bp. Prodigal solves the same problem exactly, with
# dynamic programming over start/stop nodes (Larralde 2022); greedy selection
# is a simpler approximation that works well when one gene clearly
# outscores its shadows.


def resolve_overlaps(orfs, use_coding_score=True):
    """Final non-overlapping gene set, sorted by position."""
    if use_coding_score:
        key = lambda o: o.coding_score + o.start_score
    else:
        key = lambda o: o.length            # baseline without a coding score: longest first
    genes = select_non_overlapping(orfs, key=key, max_overlap=MAX_OVERLAP)
    return sorted(genes, key=lambda o: (o.contig, o.genome_coords()))


TSV_HEADER = ["contig", "start", "end", "strand", "frame", "length_nt",
              "start_codon", "n_alt_starts", "partial",
              "rbs_motif", "rbs_spacer", "start_score", "coding_score"]


def orf_row(orf, genome):
    """Format one ORF as a list of TSV fields."""
    seq = genome[orf.contig][0 if orf.strand == "+" else 1]
    g_start, g_end = orf.genome_coords()
    start_codon = seq[orf.start:orf.start + 3] if not orf.partial5 or orf.start in orf.starts else "edge"
    # Partial code as in Prodigal: "10" = open at 5' end, "01" = open at 3' end.
    partial = f"{int(orf.partial5)}{int(orf.partial3)}"
    return [orf.contig, g_start, g_end, orf.strand, orf.frame, orf.length,
            start_codon, len(orf.starts), partial,
            orf.rbs_motif, orf.rbs_spacer, f"{orf.start_score:.2f}", f"{orf.coding_score:.1f}"]


def write_tsv(orfs, genome, path):
    with open(path, "w") as out:
        out.write("\t".join(TSV_HEADER) + "\n")
        for orf in orfs:
            out.write("\t".join(map(str, orf_row(orf, genome))) + "\n")


# =============================================================================
# Command-line interface
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict open reading frames (ORFs) in a bacterial genome.")
    parser.add_argument("fasta", help="genome in (multi-)FASTA format")
    parser.add_argument("-o", "--output", default="orfs.tsv",
                        help="output table (default: orfs.tsv)")
    parser.add_argument("--starts", default=",".join(DEFAULT_START_CODONS),
                        help="comma-separated start codons (default: ATG,GTG,TTG)")
    parser.add_argument("--gap-run", type=int, default=DEFAULT_GAP_RUN,
                        help="run of N's that breaks an ORF (default: 10)")
    parser.add_argument("--fdr", type=float, default=DEFAULT_FDR,
                        help="target false discovery rate for the length cutoff "
                             "(default: 0.5 = lenient pre-filter; e.g. 0.05 = strict)")
    parser.add_argument("--min-codons", type=int, default=None,
                        help="fixed minimum length in codons; overrides the FDR rule")
    parser.add_argument("--length-table", metavar="TSV",
                        help="write observed vs null ORF counts per length (for plotting)")
    parser.add_argument("--check-null", action="store_true",
                        help="compare the analytic null model with a shuffled genome")
    parser.add_argument("--longest-start", action="store_true",
                        help="skip start selection and keep the most upstream start "
                             "(baseline for comparison)")
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                        help="Markov order of the coding/background models (default: 5 = hexamers)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="self-training rounds for the coding model (default: 3)")
    parser.add_argument("--no-coding-score", action="store_true",
                        help="skip the coding score filter (baseline for comparison)")
    parser.add_argument("--no-overlap-filter", action="store_true",
                        help="skip overlap resolution (baseline for comparison)")
    parser.add_argument("--all-orfs", metavar="TSV",
                        help="also write every enumerated ORF candidate, before any filtering")
    args = parser.parse_args(argv)
    log = lambda msg: print(msg, file=sys.stderr)

    # --- Section 1: enumeration -------------------------------------------
    start_codons = tuple(c.strip().upper() for c in args.starts.split(","))
    orfs, genome = enumerate_orfs(args.fasta, start_codons, args.gap_run)
    n_partial = sum(o.partial5 or o.partial3 for o in orfs)
    log(f"[1] enumeration: {len(orfs)} ORF candidates "
        f"({n_partial} partial) in {len(genome)} contig(s)")
    if args.all_orfs:
        write_tsv(orfs, genome, args.all_orfs)      # "all possible ORFs", longest start each
        log(f"    all candidates written to {args.all_orfs}")

    # --- Section 2: length cutoff -----------------------------------------
    freq = base_frequencies(genome)
    genome_size = sum(len(seq) for seq, _ in genome.values())
    null = NullModel(freq, genome_size, start_codons)
    cutoff, table = fdr_length_cutoff(orfs, null, args.fdr)
    log(f"[2] GC = {freq['G'] + freq['C']:.1%}, P(stop codon) = {null.p:.4f}, "
        f"mean random stop-to-stop distance = {1 / null.p:.0f} codons")
    if args.min_codons is not None:
        log(f"    FDR rule would give {cutoff} codons; using fixed --min-codons {args.min_codons}")
        cutoff = args.min_codons
    elif cutoff < len(table) and table[cutoff][1] > 0:
        log(f"    length cutoff = {cutoff} codons (FDR <= {args.fdr:g}; "
            f"{null.expected(cutoff):.0f} random ORFs expected vs {table[cutoff][1]} observed)")
    if args.check_null:
        log("    null model check (ORFs >= L codons):  L  expected  shuffled-genome")
        for L, e, o in check_null_with_shuffle(genome, null, start_codons, args.gap_run):
            log(f"    {L:>40}  {e:>8.0f}  {o:>8}")
    if args.length_table:
        write_length_table(table, args.length_table)

    # Partial ORFs are judged by the same cutoff: their observed length is a
    # lower bound, so a long partial ORF is still strong evidence.
    orfs = [o for o in orfs if o.max_codons >= cutoff]
    log(f"    {len(orfs)} candidates pass the length cutoff")
    if not orfs:
        log("    WARNING: no ORF is long enough to stand out from the random-sequence "
            "null model; try a larger --fdr or a fixed --min-codons")

    # --- Section 3: start selection ---------------------------------------
    if args.longest_start:
        log("[3] start selection skipped (--longest-start): using the most upstream start")
    else:
        start_odds, counts = train_start_codon_odds(orfs, genome, freq, start_codons)
        n_train = sum(counts.values()) - len(counts)
        log(f"[3] start codon usage learned from {n_train} starts with a strong RBS: " +
            ", ".join(f"{c} {counts[c] / sum(counts.values()):.0%} "
                      f"(log2-odds {start_odds[c]:+.2f})" for c in start_codons))
        longest = {id(o): o.start for o in orfs}
        choose_starts(orfs, genome, start_odds, cutoff)
        moved = sum(o.start != longest[id(o)] for o in orfs)
        with_rbs = sum(o.rbs_motif != "-" for o in orfs)
        log(f"    {moved} of {len(orfs)} ORFs got a start downstream of the longest one; "
            f"{with_rbs} chosen starts have an SD motif")

    # --- Section 4: coding score ------------------------------------------
    if args.no_coding_score:
        log("[4] coding score skipped (--no-coding-score)")
    else:
        log(f"[4] coding score: order-{args.order} three-periodic Markov chain vs background, "
            f"{args.rounds} self-training round(s)")
        self_train_and_score(orfs, genome, args.order, args.rounds, log)
        orfs = [o for o in orfs if o.coding_score > 0]
        log(f"    {len(orfs)} candidates have coding log-odds > 0")

    # --- Section 5: overlaps and output -----------------------------------
    if args.no_overlap_filter:
        log("[5] overlap resolution skipped (--no-overlap-filter)")
        genes = sorted(orfs, key=lambda o: (o.contig, o.genome_coords()))
    else:
        genes = resolve_overlaps(orfs, use_coding_score=not args.no_coding_score)
        log(f"[5] {len(genes)} predicted genes after removing ORFs that overlap "
            f"a better one by > {MAX_OVERLAP} bp")
    write_tsv(genes, genome, args.output)
    log(f"    predictions written to {args.output}")


if __name__ == "__main__":
    main()
