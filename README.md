# splice-finder

`splice-finder` is a work-in-progress command-line bioinformatics project for
predicting splice donor and acceptor sites in genomic sequences using machine
learning. I am developing this project to strengthen my understanding of
bioinformatics workflows and apply concepts from my machine-learning and
Bayesian-analysis coursework.

## Project status

The project currently supports building and validating labeled donor, acceptor,
and non-site sequence datasets from a reference genome and GFF3 annotation.
Model training, evaluation, and splice-site prediction are still under
development.

## Initial organism

Initial development and testing use the
[_Saccharomyces cerevisiae_ S288C reference genome](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.1/),
available from NCBI.

The mitochondrial chromosome is excluded by default because its group I and
group II introns use different splicing mechanisms from nuclear introns.

## Installation

This project uses Conda to manage its Python environment and dependencies.

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate splicefind
```

## Input data

Place a reference genome and its corresponding annotation in `data/raw/`:

```text
data/
└── raw/
    ├── genome.fna
    └── annotation.gff3
```

The FASTA and GFF3 files must describe the same genome assembly, and
chromosome identifiers must match.

## Building the dataset

Run the dataset builder from the project root using its default paths:

```bash
python scripts/build_dataset.py
```

The default paths are:

```text
Genome:      data/raw/genome.fna
Annotation:  data/raw/annotation.gff3
Database:    data/interim/annotation.gffdb
Output:      data/processed/splice_dataset.csv
```

Custom paths and settings can be provided through command-line arguments:

```bash
python scripts/build_dataset.py \
    --genome data/raw/genome.fna \
    --annotation data/raw/annotation.gff3 \
    --database data/interim/annotation.gffdb \
    --output data/processed/splice_dataset.csv \
    --window 40 \
    --seed 42 \
    --exclude-chrom NC_001224.1
```

View all available arguments with:

```bash
python scripts/build_dataset.py --help
```

If the annotation file changes, rebuild the cached annotation database:

```bash
python scripts/build_dataset.py --rebuild-db
```

## Dataset output

The default output is:

```text
data/processed/splice_dataset.csv
```

Each row represents one genomic sequence window.

| Column         | Description                                            |
| -------------- | ------------------------------------------------------ |
| `sequence`     | DNA sequence surrounding the selected genomic position |
| `label`        | Classification label: `donor`, `acceptor`, or `none`   |
| `chrom`        | Chromosome or sequence identifier                      |
| `strand`       | Genomic strand of the example                          |
| `position`     | Genomic anchor position                                |
| `intron_start` | Start coordinate of the associated intron              |
| `intron_end`   | End coordinate of the associated intron                |

With the default window size of 40, each sequence contains 80 nucleotides.

The script generates one donor and one acceptor example for every retained
intron. It then generates as many negative examples as there are positive
examples combined. Therefore, the `none` class is twice the size of either
individual splice-site class.

## Dataset validation

During dataset construction, the script reports:

- the number of genome sequences loaded;
- feature counts from the GFF3 annotation;
- the number of inferred introns;
- excluded, duplicate, malformed, and edge-clipped examples;
- the percentage of donor windows containing `GT` near the expected position;
- the percentage of acceptor windows containing `AG` near the expected
  position; and
- final class counts.

Unexpectedly low motif percentages may indicate mismatched genome and
annotation files, an incorrect transcript feature name, or a coordinate or
strand-handling problem.

After building the dataset, the validation notebook can be run with:

```bash
cd notebooks
jupyter lab eda_validation.ipynb
```

The notebook checks sequence integrity, class balance, GC-content distributions,
positional nucleotide frequencies, and canonical splice-site motifs before
modeling.

## Reproducibility

Negative examples are generated using a configurable random seed. Running the
script with the same inputs, annotation database, settings, and seed should
produce the same sampled dataset.

The default seed is `42`. It can be changed with:

```bash
python scripts/build_dataset.py --seed 123
```

## Current limitations

- Introns are inferred only from transcripts represented by `mRNA` and `exon`
  features.
- Non-mRNA introns are not included.
- Canonical motif checks are diagnostic and do not establish that every site
  is biologically valid.
- Model training and splice-site prediction are not yet implemented.
- Repeated genomic regions can produce identical sequence windows at different
  locations. The validation notebook uses one copy of each sequence-label
  combination for sequence-level analysis.
- All negative examples are currently recorded on the plus strand, so strand
  should not be used as a model feature.

## Planned work

- Add automated tests for coordinate and strand handling.
- Create sequence features for baseline models.
- Compare statistical, machine-learning, and Bayesian approaches.
- Use chromosome- or gene-aware cross-validation.
- Evaluate donor, acceptor, and non-site predictions separately.
- Compare boosted-tree models as a stretch goal.
- Add trained-model inference to the command-line interface.
- Document model performance and dataset provenance.
