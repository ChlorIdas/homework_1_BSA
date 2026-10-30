# homework_1_BSA

Biological Sequence Analysis homework: ORF prediction in bacterial genomes and an HMM for eukaryotic gene finding.

```
homework_1_BSA/
├── README.md
├── envs_and_ORFfinder_flags/
│   ├── environment.yml          # conda environment
│   └── ORFfinder_presets.txt    # NCBI ORFfinder settings used for the comparison
├── scripts/
│   └── orf_finder.py            # ORF prediction script (the programming task)
├── sequences/                   # input genomes (NCBI RefSeq)
│   ├── GCF_000013425.1_ASM1342v1_genomic.fna   # Staphylococcus aureus NCTC 8325
│   ├── GCF_000005845.2_ASM584v2_genomic.fna    # Escherichia coli K-12 MG1655
│   └── GCF_000006765.1_ASM676v1_genomic.fna    # Pseudomonas aeruginosa PAO1
├── results/                     # Only .gitignore
│   
└── Report/
    ├── main.tex                 # report (LaTeX)
    ├── references.bib           # bibliography
    ├── figures/
    │   └── hmm_gene_model.png   # HMM state diagram
    └── suprise/                 # :)
```
