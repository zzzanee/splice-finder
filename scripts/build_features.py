"""
    File: build_features.py
    Description:
        This file uses the dataset created by build_dataset.py and creates
        a features table containing simple sequence features and positional
        one-hot encoding.

    Example:
        python scripts/build_features.py \
            --input data/processed/splice_dataset.csv \
            --output data/processed/features.csv
    
    Output:
        features.csv        
    
    See `scripts/python build_features.py --help` for options
    Date Created: 9/26/2026
    Date Last Modified: 10/8/2026   
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/processed/splice_dataset.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/features.csv")
BASES = ("A", "C", "G", "T")
EXPECTED_LABELS = {"donor", "acceptor", "none"}


def parse_args():
    """
    Parses command-line arguments for feature generation.

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    argparse.Namespace
        parsed input path, output path, and deduplication settings
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build label-independent sequence features from the splice-site "
            "dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input splice-site CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output feature CSV.",
    )
    parser.add_argument(
        "--deduplicate-sequences",
        action="store_true",
        help=(
            "Keep one row per unique sequence-label combination. By default, "
            "all genomic sites are retained and identical sequences share a "
            "sequence_group value."
        ),
    )
    return parser.parse_args()


def validate_dataset(dataframe):
    """
    Validates the sequence and label columns used for feature generation.

    PARAMETERS
    ----------
    dataframe : pandas.DataFrame
        input dataset containing sequence and label columns

    RETURNS
    -------
    int
        common sequence length

    RAISES
    ------
    ValueError
        if required columns, labels, sequence lengths, or bases are invalid
    """
    required_columns = {"sequence", "label"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    if dataframe.empty:
        raise ValueError("Input dataset is empty")
    if dataframe[["sequence", "label"]].isna().any().any():
        raise ValueError("Sequence and label columns cannot contain missing values")

    observed_labels = set(dataframe["label"].unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            "Expected labels "
            f"{sorted(EXPECTED_LABELS)}, found {sorted(observed_labels)}"
        )

    sequence_lengths = dataframe["sequence"].str.len().unique()
    if len(sequence_lengths) != 1:
        raise ValueError("All sequences must have the same length")

    sequence_length = int(sequence_lengths[0])
    if sequence_length < 2 or sequence_length % 2 != 0:
        raise ValueError(
            "Sequence length must be an even number greater than or equal to 2"
        )

    valid_sequences = dataframe["sequence"].str.fullmatch(r"[ACGT]+")
    if not valid_sequences.all():
        invalid_count = int((~valid_sequences).sum())
        raise ValueError(
            f"Found {invalid_count} sequences containing bases outside A/C/G/T"
        )

    labels_per_sequence = dataframe.groupby("sequence")["label"].nunique()
    conflicting_sequences = labels_per_sequence[labels_per_sequence > 1]
    if not conflicting_sequences.empty:
        raise ValueError(
            f"Found {len(conflicting_sequences)} identical sequences with "
            "conflicting labels"
        )

    return sequence_length


def sequence_group(sequence):
    """
    Creates a stable identifier for an exact DNA sequence.

    PARAMETERS
    ----------
    sequence : str
        DNA sequence to identify

    RETURNS
    -------
    str
        SHA-256 digest shared by identical sequences
    """
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def gc_content(sequence):
    """Calculates the proportion of G and C bases in a DNA sequence."""
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def position_name(relative_position):
    """
    Formats a relative sequence position for use in a feature name.

    PARAMETERS
    ----------
    relative_position : int
        position relative to the splice-site anchor

    RETURNS
    -------
    str
        position label such as m40 or p00
    """
    if relative_position < 0:
        return f"m{abs(relative_position):02d}"
    return f"p{relative_position:02d}"


def build_one_hot_features(sequences, window):
    """
    Creates positional one-hot features for equal-length DNA sequences.

    PARAMETERS
    ----------
    sequences : pandas.Series
        DNA sequences to encode
    window : int
        number of bases on each side of the splice-site anchor

    RETURNS
    -------
    pandas.DataFrame
        binary A, C, G, and T features for every sequence position
    """
    sequence_array = np.asarray(
        [list(sequence) for sequence in sequences],
        dtype="U1",
    )

    feature_columns = {}
    for position in range(sequence_array.shape[1]):
        relative_position = position - window
        position_label = position_name(relative_position)

        for base in BASES:
            column_name = f"pos_{position_label}_{base}"
            feature_columns[column_name] = (
                sequence_array[:, position] == base
            ).astype(int)

    return pd.DataFrame(feature_columns, index=sequences.index)


def build_features(dataframe, sequence_length, deduplicate_sequences=False):
    """
    Builds label-independent features and sequence-group metadata.

    PARAMETERS
    ----------
    dataframe : pandas.DataFrame
        validated splice-site dataset
    sequence_length : int
        common length of every DNA sequence
    deduplicate_sequences : bool, optional
        whether to retain one row per sequence-label combination, by default
        False

    RETURNS
    -------
    pandas.DataFrame
        original dataset columns, grouping metadata, targets, and features
    """
    feature_data = dataframe.copy()
    feature_data["sequence_group"] = feature_data["sequence"].apply(
        sequence_group
    )
    feature_data["site_count"] = feature_data.groupby("sequence")[
        "sequence"
    ].transform("size")

    if deduplicate_sequences:
        feature_data = (
            feature_data.drop_duplicates(subset=["sequence", "label"])
            .reset_index(drop=True)
        )

    window = sequence_length // 2
    feature_data["is_site"] = (feature_data["label"] != "none").astype(int)
    feature_data["gc_content"] = feature_data["sequence"].apply(gc_content)
    feature_data["has_donor_gt"] = (
        feature_data["sequence"].str.slice(window, window + 2).eq("GT")
    ).astype(int)
    feature_data["has_acceptor_ag"] = (
        feature_data["sequence"].str.slice(window - 1, window + 1).eq("AG")
    ).astype(int)

    one_hot_features = build_one_hot_features(
        feature_data["sequence"],
        window,
    )
    return pd.concat([feature_data, one_hot_features], axis=1)


def main():
    """
    Loads the sequence dataset, builds features, and writes the output CSV.
    """
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    dataframe = pd.read_csv(args.input)
    sequence_length = validate_dataset(dataframe)

    duplicate_count = int(
        dataframe.duplicated(subset=["sequence", "label"]).sum()
    )
    print(f"loaded {len(dataframe)} rows from {args.input}")
    print(dataframe["label"].value_counts().to_string())
    print(f"repeated sequence-label rows: {duplicate_count}")

    features = build_features(
        dataframe,
        sequence_length,
        deduplicate_sequences=args.deduplicate_sequences,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    feature_count = len(features.columns) - len(dataframe.columns)
    print(f"saved {len(features)} rows to {args.output}")
    print(f"added {feature_count} feature and grouping columns")

    if duplicate_count and not args.deduplicate_sequences:
        print(
            "note: repeated sequences were retained; use sequence_group "
            "when creating model splits"
        )


if __name__ == "__main__":
    main()
