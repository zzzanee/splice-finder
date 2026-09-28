"""
    File: build_dataset.py
    Description:
        This file collects a genome's fasta and GFF files and will create a csv
        containing splice donor/acceptor sites, and randomly selected 
        non-splice site windows 

    Example:
        python build_dataset.py \
            --genome data/raw/genome.fna \
            --annotation data/raw/annotation.gff3 \
            --output data/processed/splice_dataset.csv \
            --window 40 \
            --exclude-chrom NC_001224.1
    
    Output:
        splice_dataset.csv        
    
    See `python build_dataset.py --help` for options
    Date Created: 8/20/2026
    Date Last Modified: 9/27/2026   
"""
import argparse
import random
from pathlib import Path

import gffutils
import pandas as pd
from Bio import SeqIO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a labeled splice-site dataset from a reference genome "
            "and GFF3 annotation."
        )
    )
    parser.add_argument(
        "--genome",
        type=Path,
        default=Path("data/raw/genome.fna"),
        help="Reference genome in FASTA format.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/raw/annotation.gff3"),
        help="Genome annotation in GFF3 format.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/interim/annotation.gffdb"),
        help="Location for the cached gffutils database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/splice_dataset.csv"),
        help="Location of the output CSV file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=40,
        help="Number of bases to extract on each side of a site.",
    )
    parser.add_argument(
        "--transcript-feature",
        default="mRNA",
        help="GFF feature type representing transcripts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling negative examples.",
    )
    parser.add_argument(
        "--exclude-chrom",
        action="append",
        default=["NC_001224.1"],
        help=(
            "Chromosome or sequence ID to exclude. This option may be "
            "supplied multiple times."
        ),
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Rebuild the annotation database even if it already exists.",
    )
    return parser.parse_args()


def validate_args(args):
    """Validate command-line paths and values before processing."""
    if not args.genome.is_file():
        raise FileNotFoundError(
            f"Reference genome was not found: {args.genome}"
        )
    if not args.annotation.is_file():
        raise FileNotFoundError(
            f"Annotation file was not found: {args.annotation}"
        )
    if args.window <= 0:
        raise ValueError("--window must be greater than zero")
    if args.seed < 0:
        raise ValueError("--seed must be zero or greater")


def get_window(chrom_record, anchor_1based, strand, window):
    """
    Extracts a DNA sequence window around a 1-based genomic position and
    orients it in the direction of transcription.

    PARAMETERS
    ----------
    chrom_record : Bio.SeqRecord.SeqRecord
        chromosome sequence from which the window will be extracted
    anchor_1based : int
        1-base genomic position around which the window will be centered
    strand : str
        "+" or "-" denoting strand of splice site
    window : int
        # of bases collected either side of anchor

    RETURNS
    -------
    str
        uppercase DNA sequence containing up to twice the requested window
        size, reverse-complemented when the site is on the minus strand
    """
    if strand == "-":
        idx = anchor_1based
    else:
        idx = anchor_1based - 1

    start = max(0, idx - window)
    end = min(len(chrom_record.seq), idx + window)

    piece = chrom_record.seq[start:end]

    if strand == "-":
        piece = piece.reverse_complement()

    return str(piece).upper()


def get_introns(db, transcript_feature):
    """
    Infers intron coordinates from exon gaps of the same transcript

    PARAMETERS
    ----------
    db : gffutils.FeatureDB
        annotation database containing transcript and exon features
    transcript_feature : str
        GFF feature type used to identify transcript records

    YIELDS
    ------
    tuple
        chromosome identifier, strand, 1-based intron start, and 1-based
        intron end for each valid inferred intron
    """
    n_transcripts = 0
    n_multi_exon = 0
    n_malformed = 0

    for transcript in db.features_of_type(transcript_feature):
        n_transcripts += 1
        exons = list(
            db.children(
                transcript,
                featuretype="exon",
                order_by="start",
            )
        )

        if len(exons) < 2:
            continue
        n_multi_exon += 1

        for index in range(len(exons) - 1):
            intron_start = exons[index].end + 1
            intron_end = exons[index + 1].start - 1

            if intron_end <= intron_start:
                n_malformed += 1
                continue

            yield (
                transcript.seqid,
                transcript.strand,
                intron_start,
                intron_end,
            )

    print(
        f"  found {n_transcripts} '{transcript_feature}' features; "
        f"{n_multi_exon} have two or more exons"
    )

    if n_malformed:
        print(
            f"  skipped {n_malformed} overlapping or malformed intron gaps"
        )


def check_motif(sequences, expected, near_offset, tolerance=3):
    """
    Calculates the percentage of sequences containing an expected motif near
    a specified position.

    PARAMETERS
    ----------
    sequences : list[str]
        DNA sequences to examine
    expected : str
        nucleotide motif to search for
    near_offset : int
        expected zero-based position of the motif within each sequence
    tolerance : int
        number of positions on either side of the expected offset to search,
        by default 3

    RETURNS
    -------
    float
        percentage of sequences containing the motif within the search region
    """

    if not sequences:
        print(f"  no sequences available to check for '{expected}'")
        return 0.0

    hits = 0

    for sequence in sequences:
        start = max(0, near_offset - tolerance)
        end = min(
            len(sequence),
            near_offset + tolerance + len(expected),
        )
        neighborhood = sequence[start:end]

        if expected in neighborhood:
            hits += 1

    percentage = 100 * hits / len(sequences)

    print(
        f"  {hits}/{len(sequences)} ({percentage:.1f}%) contain "
        f"'{expected}' near position {near_offset}"
    )

    return percentage


def load_genome(genome_path):
    """
    Loads all sequences from a reference genome FASTA file.

    PARAMETERS
    ----------
    genome_path : pathlib.Path
        path to the reference genome FASTA file

    RETURNS
    -------
    dict[str, Bio.SeqRecord.SeqRecord]
        dictionary mapping each FASTA sequence identifier to its sequence
        record

    RAISES
    ------
    ValueError
        if the FASTA file contains no sequence records
    """

    print("loading genome...")

    genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

    if not genome:
        raise ValueError(
            f"No FASTA records were found in {genome_path}"
        )

    preview = list(genome.keys())[:5]
    suffix = " ..." if len(genome) > 5 else ""

    print(
        f"  {len(genome)} sequences loaded: {preview}{suffix}"
    )

    return genome


def load_annotation(annotation_path, database_path, rebuild_db):
    """
    Loads a cached annotation database or creates one from a GFF3 annotation
    file.

    PARAMETERS
    ----------
    annotation_path : pathlib.Path
        path to the source GFF3 annotation file
    database_path : pathlib.Path
        path at which the gffutils database is stored
    rebuild_db : bool
        whether to recreate the database when a cached database already
        exists

    RETURNS
    -------
    gffutils.FeatureDB
        searchable database containing features from the GFF3 annotation
    """

    print("loading annotation...")

    if database_path.exists() and not rebuild_db:
        print(f"  using cached database: {database_path}")
        db = gffutils.FeatureDB(str(database_path))
    else:
        database_path.parent.mkdir(parents=True, exist_ok=True)

        if rebuild_db and database_path.exists():
            print(f"  rebuilding database: {database_path}")
        else:
            print(f"  creating database: {database_path}")

        db = gffutils.create_db(
            str(annotation_path),
            dbfn=str(database_path),
            force=True,
            keep_order=True,
            merge_strategy="create_unique",
        )

    print("  feature types in annotation:")

    for feature_type in db.featuretypes():
        count = db.count_features_of_type(feature_type)
        print(f"    {feature_type}: {count}")

    return db


def extract_positive_examples(
    genome,
    db,
    transcript_feature,
    excluded_chroms,
    window,
):
    """
    Extracts donor and acceptor sequence windows from annotated introns while
    removing duplicate splice sites and unusable windows.

    PARAMETERS
    ----------
    genome : dict[str, Bio.SeqRecord.SeqRecord]
        dictionary mapping chromosome identifiers to genome sequence records
    db : gffutils.FeatureDB
        annotation database containing transcript and exon features
    transcript_feature : str
        GFF feature type used to identify transcript records
    excluded_chroms : set[str]
        chromosome identifiers that should not contribute examples
    window : int
        number of bases to extract on each side of each splice site

    RETURNS
    -------
    tuple[list[dict], set[tuple]]
        dataset rows for unique donor and acceptor examples, followed by the
        chromosome-position pairs that must be avoided during negative
        sampling
    """

    rows = []
    used_positions = set()
    seen_sites = set()

    n_introns = 0
    n_excluded = 0
    n_missing_chrom = 0
    n_edge_clipped = 0
    n_duplicates = 0

    print("walking transcripts for introns...")

    for seqid, strand, intron_start, intron_end in get_introns(
        db,
        transcript_feature,
    ):
        n_introns += 1

        if seqid not in genome:
            n_missing_chrom += 1
            continue

        if seqid in excluded_chroms:
            n_excluded += 1
            continue

        chrom = genome[seqid]

        if strand == "+":
            donor_anchor = intron_start
            acceptor_anchor = intron_end
        elif strand == "-":
            donor_anchor = intron_end
            acceptor_anchor = intron_start
        else:
            print(
                f"  warning: skipping intron with unknown strand "
                f"'{strand}' on {seqid}"
            )
            continue

        sites = [
            ("donor", donor_anchor),
            ("acceptor", acceptor_anchor),
        ]

        for label, anchor in sites:
            site_key = (seqid, strand, anchor, label)

            if site_key in seen_sites:
                n_duplicates += 1
                continue

            seen_sites.add(site_key)

            sequence = get_window(
                chrom_record=chrom,
                anchor_1based=anchor,
                strand=strand,
                window=window,
            )

            if len(sequence) != 2 * window:
                n_edge_clipped += 1
                continue

            rows.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "chrom": seqid,
                    "strand": strand,
                    "position": anchor,
                    "intron_start": intron_start,
                    "intron_end": intron_end,
                }
            )

        used_positions.add((seqid, intron_start))
        used_positions.add((seqid, intron_end))

    print(f"  examined {n_introns} annotated introns")
    print(f"  extracted {len(rows)} unique positive windows")
    print(f"  skipped {n_excluded} introns from excluded chromosomes")
    print(f"  skipped {n_missing_chrom} introns with missing chromosomes")
    print(f"  skipped {n_edge_clipped} windows near chromosome edges")
    print(f"  skipped {n_duplicates} duplicate splice sites")

    return rows, used_positions


def generate_negative_examples(
    genome,
    excluded_chroms,
    used_positions,
    count,
    window,
    rng,
):
    """
    Generates random non-splice-site sequence windows while avoiding annotated
    splice boundaries and duplicate sampling positions.

    PARAMETERS
    ----------
    genome : dict[str, Bio.SeqRecord.SeqRecord]
        dictionary mapping chromosome identifiers to genome sequence records
    excluded_chroms : set[str]
        chromosome identifiers that should not contribute examples
    used_positions : set[tuple]
        chromosome-position pairs representing annotated splice boundaries
    count : int
        number of negative examples to generate
    window : int
        number of bases to extract on each side of each sampled position
    rng : random.Random
        initialized random-number generator used for reproducible sampling

    RETURNS
    -------
    list[dict]
        dataset rows containing randomly sampled negative examples

    RAISES
    ------
    ValueError
        if no chromosome is long enough to provide a complete sequence window
    RuntimeError
        if the requested number of negative examples cannot be generated
        within the maximum number of attempts
    """

    print("generating negative examples...")

    chromosome_ids = [
        seqid
        for seqid, record in genome.items()
        if seqid not in excluded_chroms
        and len(record.seq) > 2 * window + 2
    ]

    if not chromosome_ids:
        raise ValueError(
            "No chromosomes are long enough for negative sampling"
        )

    # Sampling weights make positions on longer chromosomes more likely.
    chromosome_weights = [
        len(genome[seqid].seq) - 2 * window
        for seqid in chromosome_ids
    ]

    rows = []
    selected_positions = set()
    attempts = 0
    max_attempts = count * 50

    while len(rows) < count and attempts < max_attempts:
        attempts += 1

        seqid = rng.choices(
            chromosome_ids,
            weights=chromosome_weights,
            k=1,
        )[0]

        chrom = genome[seqid]
        chrom_len = len(chrom.seq)

        position = rng.randint(
            window + 1,
            chrom_len - window - 1,
        )

        position_key = (seqid, position)

        if position_key in selected_positions:
            continue

        near_splice_site = any(
            (seqid, nearby_position) in used_positions
            for nearby_position in range(
                position - window,
                position + window + 1,
            )
        )

        if near_splice_site:
            continue

        sequence = get_window(
            chrom_record=chrom,
            anchor_1based=position,
            strand="+",
            window=window,
        )

        if len(sequence) != 2 * window:
            continue

        selected_positions.add(position_key)

        rows.append(
            {
                "sequence": sequence,
                "label": "none",
                "chrom": seqid,
                "strand": "+",
                "position": position,
                "intron_start": None,
                "intron_end": None,
            }
        )

    print(
        f"  generated {len(rows)} negative examples "
        f"after {attempts} attempts"
    )

    if len(rows) < count:
        raise RuntimeError(
            f"Only generated {len(rows)} of the requested {count} "
            f"negative examples after {attempts} attempts. "
            "Try reducing the window size or reviewing the excluded regions."
        )

    return rows


def save_dataset(rows, output_path):
    """
    Converts dataset rows into a data frame and saves them as a CSV file.

    PARAMETERS
    ----------
    rows : list[dict]
        donor, acceptor, and negative dataset records to save
    output_path : pathlib.Path
        path at which the completed CSV file will be written
    
    RAISES
    ------
    ValueError
        if supplied collection of dataset rows is empty
    """

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError("Refusing to save an empty dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print(f"saved {len(dataframe)} rows to {output_path}")
    print("label counts:")
    print(dataframe["label"].value_counts().to_string())


def main(args):
    """
    Runs the complete pipeline for loading genomic data, extracting splice
    sites, sampling negative examples, and saving the finished dataset.

    PARAMETERS
    ----------
    args : argparse.Namespace
        command-line arguments given or default arguments
    """

    validate_args(args)

    rng = random.Random(args.seed)
    excluded_chroms = set(args.exclude_chrom)

    genome = load_genome(args.genome)

    db = load_annotation(
        annotation_path=args.annotation,
        database_path=args.database,
        rebuild_db=args.rebuild_db,
    )

    positive_rows, used_positions = extract_positive_examples(
        genome=genome,
        db=db,
        transcript_feature=args.transcript_feature,
        excluded_chroms=excluded_chroms,
        window=args.window,
    )

    if not positive_rows:
        raise ValueError(
            "No positive splice-site examples were extracted. "
            "Check the transcript feature name, chromosome IDs, and annotation."
        )

    print("checking canonical splice motifs:")

    donor_sequences = [
        row["sequence"]
        for row in positive_rows
        if row["label"] == "donor"
    ]

    acceptor_sequences = [
        row["sequence"]
        for row in positive_rows
        if row["label"] == "acceptor"
    ]

    donor_percentage = check_motif(
        donor_sequences,
        expected="GT",
        near_offset=args.window,
    )

    acceptor_percentage = check_motif(
        acceptor_sequences,
        expected="AG",
        near_offset=args.window,
    )

    if donor_percentage < 90 or acceptor_percentage < 90:
        print(
            "  warning: canonical motif percentages are below 90%. "
            "Review coordinate and strand handling before training a model."
        )

    negative_rows = generate_negative_examples(
        genome=genome,
        excluded_chroms=excluded_chroms,
        used_positions=used_positions,
        count=len(positive_rows),
        window=args.window,
        rng=rng,
    )

    all_rows = positive_rows + negative_rows

    # Shuffle deterministically so the CSV is not grouped entirely by label.
    rng.shuffle(all_rows)

    save_dataset(
        rows=all_rows,
        output_path=args.output,
    )


if __name__ == "__main__":
    main(parse_args())