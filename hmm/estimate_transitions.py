#!/usr/bin/env python3
"""
estimate_transitions.py - estimate the gene-finding HMM's transition probabilities by
counting in an annotated genome (training from labelled sequences, Durbin et al. 3.3).

Every base of an annotated genome already has its label (coding exon, intron,
intergenic), so maximum-likelihood transition probabilities are simple counts:
  * self-loop of a region  p = 1 - 1/(mean length)       (mean of a geometric length)
  * exon -> intron (phase k) = introns of phase k / codons read
  * exon -> stop             = genes / codons read
  * stop-codon branches      = counts of TAA, TAG, TGA

Model conventions (see the HMM figure):
  * Exons are coding parts only (CDS). UTRs count as intergenic, since the model has
    no UTR states.
  * One transcript per gene: the one with the longest coding sequence.
  * Intron phase = number of coding bases before the intron, modulo 3:
    1 = after the 1st codon base (E1 -> I1), 2 = after the 2nd (E2 -> I2),
    0 = between codons (E3 -> I0).
  * Intron "body" = intron length minus the 2 donor (GT) and 2 acceptor (AG) states.

Usage: python estimate_transitions.py genomic.fna.gz genomic.gff.gz [--chrom NC_003279.8]
"""
import argparse
import gzip
from collections import Counter, defaultdict

from Bio import SeqIO
from Bio.Seq import reverse_complement


def open_any(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def parse_attributes(text):
    return dict(kv.split("=", 1) for kv in text.strip().split(";") if "=" in kv)


def read_coding_genes(gff_path, chrom):
    """Return {gene_id: (strand, sorted CDS segments)} using the longest CDS transcript."""
    gene_of_rna, coding_genes = {}, set()
    cds = defaultdict(list)                      # transcript id -> [(start, end)]
    strand_of = {}
    with open_any(gff_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9 or f[0] != chrom:
                continue
            kind, start, end, strand, attrs = f[2], int(f[3]), int(f[4]), f[6], parse_attributes(f[8])
            if kind == "gene" and attrs.get("gene_biotype") == "protein_coding":
                coding_genes.add(attrs["ID"])
            elif kind == "mRNA":
                gene_of_rna[attrs["ID"]] = attrs.get("Parent")
            elif kind == "CDS" and "Parent" in attrs:
                cds[attrs["Parent"]].append((start, end))
                strand_of[attrs["Parent"]] = strand

    best = {}                                    # gene -> (cds_length, strand, segments)
    for rna, segs in cds.items():
        gene = gene_of_rna.get(rna)
        if gene not in coding_genes:
            continue
        segs = sorted(set(segs))
        length = sum(e - s + 1 for s, e in segs)
        if length % 3 == 0 and (gene not in best or length > best[gene][0]):
            best[gene] = (length, strand_of[rna], segs)
    return {g: (strand, segs) for g, (_, strand, segs) in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta"); ap.add_argument("gff")
    ap.add_argument("--chrom", default="NC_003279.8", help="chromosome to use (default C. elegans chr I)")
    args = ap.parse_args()

    seq = None
    with open_any(args.fasta) as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if rec.id == args.chrom:
                seq = str(rec.seq).upper()
                break
    if seq is None:
        raise SystemExit(f"chromosome {args.chrom} not found in {args.fasta}")
    genes = read_coding_genes(args.gff, args.chrom)

    codons = 0                                   # sense codons read by E1-E3 (no start, no stop)
    phase = Counter()                            # intron phase -> count (internal introns only)
    skipped_introns = 0                          # introns inside the start or stop codon (not in model)
    intron_lengths, stops, starts, splice = [], Counter(), Counter(), Counter()
    spans = []
    for strand, segs in genes.values():
        spans.append((segs[0][0], segs[-1][1]))
        # coding sequence 5'->3' on the gene's strand
        order = segs if strand == "+" else segs[::-1]
        if strand == "+":
            cds_seq = "".join(seq[s - 1:e] for s, e in order)
        else:
            cds_seq = "".join(reverse_complement(seq[s - 1:e]) for s, e in order)
        starts[cds_seq[:3]] += 1
        stops[cds_seq[-3:]] += 1
        n_sense = len(cds_seq) // 3 - 2          # minus start and stop codon
        codons += n_sense

        before = 0                               # coding bases before the current intron
        for (s1, e1), (s2, e2) in zip(order, order[1:]):
            before += e1 - s1 + 1
            if strand == "+":
                intron = seq[e1:s2 - 1]
            else:
                intron = reverse_complement(seq[e2:s1 - 1])
            splice[(intron[:2], intron[-2:])] += 1
            intron_lengths.append(len(intron))
            if before <= 3 or before >= len(cds_seq) - 3:
                skipped_introns += 1             # inside/next to start or stop codon
            else:
                phase[before % 3] += 1

    spans.sort()
    gaps, reach = [], 0
    for s, e in spans:                           # intergenic = gaps between coding spans
        if reach and s > reach + 1:
            gaps.append(s - reach - 1)
        reach = max(reach, e)

    n_genes = len(genes)
    mean = lambda xs: sum(xs) / len(xs)
    body = mean(intron_lengths) - 4
    p = {
        "N -> N": 1 - 1 / mean(gaps),
        "N -> start": 1 / mean(gaps),
        "E1 -> I1 (phase 1)": phase[1] / codons,
        "E1 -> E2": 1 - phase[1] / codons,
        "E2 -> I2 (phase 2)": phase[2] / codons,
        "E2 -> E3": 1 - phase[2] / codons,
        "E3 -> I0 (phase 0)": phase[0] / codons,
        "E3 -> stop": n_genes / codons,
        "E3 -> E1": 1 - (phase[0] + n_genes) / codons,
        "intron body loop": 1 - 1 / body,
        "intron body -> acceptor": 1 / body,
    }
    taa, tag, tga = stops["TAA"], stops["TAG"], stops["TGA"]
    total_stops = taa + tag + tga

    print(f"chromosome {args.chrom}: {n_genes} protein-coding genes, {codons} sense codons, "
          f"{len(intron_lengths)} coding introns ({skipped_introns} next to start/stop not modelled)")
    print(f"mean lengths: intergenic {mean(gaps):.0f} bp, intron {mean(intron_lengths):.0f} bp, "
          f"gene {codons / n_genes:.0f} sense codons")
    print(f"intron phases: 0 = {phase[0]}, 1 = {phase[1]}, 2 = {phase[2]}")
    gtag = splice[("GT", "AG")] / sum(splice.values())
    print(f"GT...AG introns: {gtag:.1%};  ATG starts: {starts['ATG'] / n_genes:.1%}")
    print("\nTransition probabilities")
    for k, v in p.items():
        print(f"  {k:<26} {v:.4f}")
    print("\nStop codon branches")
    print(f"  T -> A   {(taa + tag) / total_stops:.2f}   (TAA {taa}, TAG {tag})")
    print(f"  T -> G   {tga / total_stops:.2f}   (TGA {tga})")
    print(f"  TA -> A  {taa / (taa + tag):.2f}")
    print(f"  TA -> G  {tag / (taa + tag):.2f}")
    print(f"  ({total_stops / n_genes:.1%} of genes end in TAA/TAG/TGA)")


if __name__ == "__main__":
    main()
