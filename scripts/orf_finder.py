#!/usr/bin/env python3
"""
orf_finder_simple.py - predict protein-coding ORFs in a bacterial genome.

Steps:
  1. Find all ORFs in the six reading frames (start: ATG/GTG/TTG, stop: TAA/TAG/TGA),
     keeping the longest ORF for each stop codon.
  2. Remove ORFs shorter than a genome-specific minimum length, computed from a
     random-sequence (null) model with the genome's own base composition.
  3. Score the remaining ORFs with a codon-usage log-odds score (coding vs random)
     and keep those with a positive score.
  4. Remove ORFs that overlap a higher-scoring ORF by more than 60 bp.

Usage:  python orf_finder_simple.py genome.fasta -o predicted_orfs.tsv
Requires Biopython.
"""

import argparse
import math
from collections import Counter

from Bio import SeqIO
from Bio.Seq import reverse_complement

STARTS = {"ATG", "GTG", "TTG"}          # bacterial start codons (NCBI table 11)
STOPS = {"TAA", "TAG", "TGA"}
SENSE = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in STOPS]


# ---------------------------------------------------------------------------
# 1. ORF enumeration
# ---------------------------------------------------------------------------

def orfs_in_strand(seq):
    """Yield (start, end) of ORFs in one strand (0-based, end includes the stop codon).

    Walking codon by codon in each frame, we remember the first start codon seen
    since the last stop; when the next stop arrives, that start gives the longest ORF.
    """
    for frame in range(3):
        first_start = None
        for i in range(frame, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if codon in STOPS:
                if first_start is not None:
                    yield first_start, i + 3
                first_start = None
            elif codon in STARTS and first_start is None:
                first_start = i


def find_orfs(fasta):
    """Return all ORFs in all contigs as dicts, plus the contig sequences."""
    orfs, contigs = [], {}
    for record in SeqIO.parse(fasta, "fasta"):
        seq = str(record.seq).upper()
        contigs[record.id] = seq
        L = len(seq)
        for strand, s in (("+", seq), ("-", reverse_complement(seq))):
            for start, end in orfs_in_strand(s):
                # convert to 1-based forward-strand coordinates
                if strand == "+":
                    left, right = start + 1, end
                else:
                    left, right = L - end + 1, L - start
                orfs.append({"contig": record.id, "strand": strand, "left": left,
                             "right": right, "seq": s[start:end]})
    return orfs, contigs


# ---------------------------------------------------------------------------
# 2. Genome-specific minimum length (null model)
# ---------------------------------------------------------------------------

def base_frequencies(contigs):
    """Base frequencies over both strands (so f(A)=f(T), f(C)=f(G))."""
    counts = Counter()
    for seq in contigs.values():
        counts.update(seq)
    at = counts["A"] + counts["T"]
    gc = counts["C"] + counts["G"]
    total = at + gc
    return {"A": at / total / 2, "T": at / total / 2, "C": gc / total / 2, "G": gc / total / 2}


def codon_prob(codon, freq):
    return freq[codon[0]] * freq[codon[1]] * freq[codon[2]]


def min_length(freq, alpha):
    """Length (codons) exceeded by only a fraction alpha of random ORFs.

    In random sequence each codon is a stop with probability p, so ORF length is
    geometric: P(length >= L) = (1 - p)^L. Solve (1 - p)^L = alpha for L.
    """
    p_stop = sum(codon_prob(c, freq) for c in STOPS)
    return math.ceil(math.log(alpha) / math.log(1 - p_stop)), p_stop


# ---------------------------------------------------------------------------
# 3. Codon-usage log-odds score
# ---------------------------------------------------------------------------

def codons(orf_seq):
    """Sense codons of an ORF, without the start and stop codon."""
    return [orf_seq[i:i + 3] for i in range(3, len(orf_seq) - 3, 3)]


def train_codon_log_odds(training_orfs, freq):
    """log2( P(codon | coding) / P(codon | random) ) for every sense codon.

    P(codon | coding): codon frequencies in the training ORFs (+1 pseudocount).
    P(codon | random): from base frequencies, renormalised over sense codons.
    """
    counts = Counter()
    for orf in training_orfs:
        counts.update(codons(orf["seq"]))
    n = sum(counts[c] + 1 for c in SENSE)
    random_total = sum(codon_prob(c, freq) for c in SENSE)
    return {c: math.log2(((counts[c] + 1) / n) / (codon_prob(c, freq) / random_total))
            for c in SENSE}


def score(orf_seq, log_odds):
    """Total log-odds; > 0 means the ORF looks more like coding than random DNA."""
    return sum(log_odds.get(c, 0.0) for c in codons(orf_seq))   # codons with N score 0


# ---------------------------------------------------------------------------
# 4. Overlap filter
# ---------------------------------------------------------------------------

def remove_overlaps(orfs, max_overlap):
    """Keep ORFs best score first; drop any that overlaps a kept ORF by > max_overlap bp.

    Removes "shadow" ORFs: the same DNA read in another frame or on the other strand.
    """
    kept = []
    for o in sorted(orfs, key=lambda o: o["score"], reverse=True):
        if all(k["contig"] != o["contig"] or
               min(k["right"], o["right"]) - max(k["left"], o["left"]) + 1 <= max_overlap
               for k in kept):
            kept.append(o)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict ORFs in a bacterial genome.")
    parser.add_argument("fasta", help="genome in FASTA format")
    parser.add_argument("-o", "--output", default="predicted_orfs.tsv")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="fraction of random ORFs allowed above the length cutoff (default 0.01)")
    parser.add_argument("--train-top", type=float, default=0.5,
                        help="fraction of longest ORFs used to train the codon model (default 0.5)")
    parser.add_argument("--max-overlap", type=int, default=60,
                        help="largest overlap (bp) allowed between two predicted ORFs (default 60)")
    args = parser.parse_args()

    # 1. all ORFs
    orfs, contigs = find_orfs(args.fasta)
    print(f"1. ORFs found in six frames: {len(orfs)}")

    # 2. length filter
    freq = base_frequencies(contigs)
    cutoff, p_stop = min_length(freq, args.alpha)
    long_orfs = [o for o in orfs if (len(o["seq"]) - 3) // 3 >= cutoff]
    print(f"2. GC content {freq['G'] + freq['C']:.1%}, P(stop) = {p_stop:.4f} -> "
          f"minimum length {cutoff} codons; {len(long_orfs)} ORFs pass")

    # 3. codon-usage score, trained on the longest ORFs (most likely real genes)
    long_orfs.sort(key=lambda o: len(o["seq"]), reverse=True)
    training = long_orfs[:max(1, int(len(long_orfs) * args.train_top))]
    log_odds = train_codon_log_odds(training, freq)
    for o in long_orfs:
        o["score"] = score(o["seq"], log_odds)
    predicted = [o for o in long_orfs if o["score"] > 0]
    print(f"3. codon model trained on {len(training)} longest ORFs; "
          f"{len(predicted)} ORFs have a positive coding score")

    # 4. overlap filter
    predicted = remove_overlaps(predicted, args.max_overlap)
    print(f"4. {len(predicted)} ORFs left after removing overlaps > {args.max_overlap} bp")

    # output, sorted by position
    predicted.sort(key=lambda o: (o["contig"], o["left"]))
    with open(args.output, "w") as out:
        out.write("contig\tstart\tend\tstrand\tlength_nt\tstart_codon\tscore\n")
        for o in predicted:
            out.write(f"{o['contig']}\t{o['left']}\t{o['right']}\t{o['strand']}\t"
                      f"{len(o['seq'])}\t{o['seq'][:3]}\t{o['score']:.1f}\n")
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
