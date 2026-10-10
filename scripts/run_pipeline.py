"""
    File: run_pipeline.py
    Description:
        Runs the splice-finder dataset, feature, and modeling workflow.

    Example:
        python scripts/run_pipeline.py \
            --genome data/raw/genome.fna \
            --annotation data/raw/annotation.gff3 \
            --output-dir results/species \
            --exclude-chrom NC_001328.1
    
    Output:
        results/species/
        ├── data/
        │   ├── splice_dataset.csv
        │   └── features.csv
        ├── metrics/
        │   ├── chromosome_holdout_folds.csv
        │   ├── chromosome_holdout_summary.csv
        │   └── run_information.json
        ├── models/
        │  └── multiclass_models.joblib      
        └── figures/
            ├── macro_f1_by_fold.png
            ├── confusion_matrix_logistic_regression.png
            └── confusion_matrix_xgboost.png
    
    Use `scripts/python run_pipeline.py --help` for options
    Date Created: 10/7/2026
    Date Last Modified: 10/8/2026   
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


CORE_FEATURE_COLUMNS = [
    "gc_content",
    "has_donor_gt",
    "has_acceptor_ag",
]
LABEL_ORDER = ["none", "donor", "acceptor"]


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Paths and settings used by the pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build splice-site data and features, evaluate multiclass models "
            "by chromosome, and save models trained on all rows."
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
        required=True,
        help="Genome annotation in GFF3 format.",    
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Where all pipeline outputs will be stored."    
    )
    parser.add_argument(
        "--window", 
        type=int, 
        default=40,
        help="Number of bases to extract on each side of a site.",    
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"    
    )
    parser.add_argument(
        "--hard-negative-fraction", 
        type=float, 
        default=0.5,
        help=(
            "Fraction of negative examples centered on genomic GT or AG "
            "motifs. The targeted examples are divided equally between "
            "donor-like GT and acceptor-like AG motifs."
        )    
    )
    parser.add_argument(
        "--transcript-feature", 
        default="mRNA",
        help="Tells which GFF3 feature represents transcripts."    
    )
    parser.add_argument(
        "--exclude-chrom",
        action="append",
        default=[],
        help="Sequence ID to exclude; may be supplied more than once.",
    )
    parser.add_argument(
        "--folds", 
        type=int, 
        default=5,
        help=(
            "Controls how many chromosome groups are used in the analysis. "
            "Number cannot exceed the number of chromosomes."
        )    
    )
    parser.add_argument(
        "--jobs", 
        type=int, 
        default=-1,
        help="How many CPU cores are allotted to XGBoost calculation"    
    )
    parser.add_argument(
        "--rebuild-db", 
        action="store_true",
        help="Rebuild the annotation database even if it already exists.",    
    )
    parser.add_argument(
        "--deduplicate-sequences",
        action="store_true",
        help="Pass sequence deduplication to build_features.py.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse the dataset CSV already present in the output directory.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Reuse features.csv already present in the output directory.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only build the dataset and feature table.",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Train only multiclass logistic regression.",
    )
    return parser.parse_args()


def validate_args(args):
    """
    Validate pipeline paths and numeric settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    FileNotFoundError
        If an input file or required script cannot be found.
    ValueError
        If a numeric setting is outside its allowed range.
    """
    if not args.skip_data:
        for path in (args.genome, args.annotation):
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {path}")
    if args.window <= 0:
        raise ValueError("--window must be greater than zero")
    if args.seed < 0:
        raise ValueError("--seed must be zero or greater")
    if not 0 <= args.hard_negative_fraction <= 1:
        raise ValueError("--hard-negative-fraction must be between 0 and 1")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")


def run_command(command, stage_name):
    """
    Run one pipeline stage and stop if it fails.

    Parameters
    ----------
    command : list[str]
        Command and arguments to execute.
    stage_name : str
        Human-readable name printed before execution.
    """
    print(f"\n[{stage_name}]")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def build_chromosome_folds(dataframe, split_column, n_splits):
    """
    Assign chromosomes to approximately equal-sized folds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Feature table containing chromosome identifiers.
    split_column : str
        Name of the chromosome or contig column.
    n_splits : int
        Requested number of folds.

    Returns
    -------
    list[list[str]]
        Chromosome identifiers assigned to each fold.
    """
    chromosome_sizes = (
        dataframe.groupby(split_column).size().sort_values(ascending=False)
    )
    fold_count = min(n_splits, len(chromosome_sizes))
    if fold_count < 2:
        raise ValueError("At least two chromosomes or contigs are required")

    folds = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    for chromosome, size in chromosome_sizes.items():
        smallest_fold = int(np.argmin(fold_sizes))
        folds[smallest_fold].append(chromosome)
        fold_sizes[smallest_fold] += int(size)
    return folds


def make_models(seed, jobs, include_xgboost):
    """
    Create fresh multiclass model instances.

    Parameters
    ----------
    seed : int
        Random seed used by the models.
    jobs : int
        Number of processor threads used by XGBoost.
    include_xgboost : bool
        Whether to create an XGBoost model.

    Returns
    -------
    dict[str, object]
        Model names mapped to unfitted estimators.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            random_state=seed,
        )
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except Exception as error:
            print(
                "warning: XGBoost could not be imported; continuing without "
                f"it\n  {type(error).__name__}: {error}\n"
                "  Check that xgboost and its required system libraries are "
                "installed in the active environment."
            )
        else:
            models["xgboost"] = XGBClassifier(
                objective="multi:softprob",
                num_class=len(LABEL_ORDER),
                n_estimators=150,
                max_depth=2,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=jobs,
            )
    return models


def get_modeling_columns(dataframe):
    """
    Identify split and predictor columns in the feature table.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Feature table produced by build_features.py.

    Returns
    -------
    tuple[str, list[str]]
        Split-column name followed by predictor-column names.

    Raises
    ------
    ValueError
        If required metadata, labels, or predictors are missing.
    """
    split_column = "chrom" if "chrom" in dataframe.columns else "contig"
    required = {split_column, "label", *CORE_FEATURE_COLUMNS}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"features.csv is missing columns: {sorted(missing)}")

    observed_labels = set(dataframe["label"].unique())
    if observed_labels != set(LABEL_ORDER):
        raise ValueError(
            f"Expected labels {LABEL_ORDER}, found {sorted(observed_labels)}"
        )

    position_columns = [
        column for column in dataframe.columns if column.startswith("pos_")
    ]
    if not position_columns:
        raise ValueError("No positional one-hot features were found")
    return split_column, CORE_FEATURE_COLUMNS + position_columns


def standardize(train, test, feature_columns):
    """
    Standardize predictors using training-fold statistics only.

    Parameters
    ----------
    train : pandas.DataFrame
        Training rows.
    test : pandas.DataFrame
        Test rows.
    feature_columns : list[str]
        Predictor columns to standardize.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        Standardized training and test predictors, means, and standard
        deviations.
    """
    means = train[feature_columns].mean()
    standard_deviations = train[feature_columns].std().replace(0, 1)
    x_train = (train[feature_columns] - means) / standard_deviations
    x_test = (test[feature_columns] - means) / standard_deviations
    return x_train, x_test, means, standard_deviations


def evaluate_models(
    dataframe,
    split_column,
    feature_columns,
    folds,
    seed,
    jobs,
    include_xgboost,
):
    """
    Evaluate models using chromosome-held-out folds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Complete feature table.
    split_column : str
        Chromosome or contig column used for splitting.
    feature_columns : list[str]
        Predictor columns used by each model.
    folds : list[list[str]]
        Chromosome identifiers assigned to test folds.
    seed : int
        Random seed used by the models.
    jobs : int
        Processor threads used by XGBoost.
    include_xgboost : bool
        Whether to evaluate XGBoost.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, dict[str, numpy.ndarray]]]
        Fold-level metrics followed by held-out true and predicted labels for
        each model.
    """
    label_to_integer = {
        label: index for index, label in enumerate(LABEL_ORDER)
    }
    results = []
    held_out_predictions = {}

    for fold_number, held_out in enumerate(folds, start=1):
        print(f"\n[training fold {fold_number}/{len(folds)}: {held_out}]")
        test = dataframe[dataframe[split_column].isin(held_out)].copy()
        train = dataframe[~dataframe[split_column].isin(held_out)].copy()

        overlap_count = 0
        if "sequence_group" in dataframe.columns:
            test_groups = set(test["sequence_group"])
            overlap_mask = train["sequence_group"].isin(test_groups)
            overlap_count = int(overlap_mask.sum())
            train = train[~overlap_mask].copy()

        for name, fold_data in (("training", train), ("test", test)):
            missing_labels = set(LABEL_ORDER).difference(
                fold_data["label"].unique()
            )
            if missing_labels:
                raise ValueError(
                    f"Fold {fold_number} {name} rows are missing labels: "
                    f"{sorted(missing_labels)}"
                )

        x_train, x_test, _, _ = standardize(
            train,
            test,
            feature_columns,
        )
        y_train = train["label"].map(label_to_integer).to_numpy()
        y_test = test["label"].map(label_to_integer).to_numpy()

        models = make_models(seed, jobs, include_xgboost)
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            model_predictions = held_out_predictions.setdefault(
                model_name,
                {"true": [], "predicted": []},
            )
            model_predictions["true"].append(y_test)
            model_predictions["predicted"].append(predictions)
            results.append(
                {
                    "fold": fold_number,
                    "held_out_chromosomes": ", ".join(held_out),
                    "model": model_name,
                    "training_rows": len(train),
                    "test_rows": len(test),
                    "removed_training_overlaps": overlap_count,
                    "accuracy": accuracy_score(y_test, predictions),
                    "macro_f1": f1_score(
                        y_test,
                        predictions,
                        average="macro",
                    ),
                }
            )
    combined_predictions = {
        model_name: {
            key: np.concatenate(arrays)
            for key, arrays in prediction_sets.items()
        }
        for model_name, prediction_sets in held_out_predictions.items()
    }
    return pd.DataFrame(results), combined_predictions


def save_evaluation_outputs(
    fold_results,
    summary,
    held_out_predictions,
    metrics_dir,
    figures_dir,
):
    """
    Save evaluation tables and figures.

    Parameters
    ----------
    fold_results : pandas.DataFrame
        Fold-level accuracy and macro-F1 results.
    summary : pandas.DataFrame
        Metrics summarized across chromosome folds.
    held_out_predictions : dict[str, dict[str, numpy.ndarray]]
        True and predicted labels collected from held-out rows.
    metrics_dir : pathlib.Path
        Directory for CSV metric tables.
    figures_dir : pathlib.Path
        Directory for PNG figures.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(
        metrics_dir / "chromosome_holdout_folds.csv",
        index=False,
    )
    summary.to_csv(
        metrics_dir / "chromosome_holdout_summary.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for model_name, model_results in fold_results.groupby("model"):
        axis.plot(
            model_results["fold"],
            model_results["macro_f1"],
            marker="o",
            linewidth=2,
            label=model_name.replace("_", " ").title(),
        )
    axis.set(
        xlabel="Chromosome fold",
        ylabel="Macro F1",
        title="Macro F1 Across Chromosome-Held-Out Folds",
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figures_dir / "macro_f1_by_fold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    class_metric_rows = []
    for model_name, prediction_sets in held_out_predictions.items():
        true_labels = prediction_sets["true"]
        predicted_labels = prediction_sets["predicted"]
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=range(len(LABEL_ORDER)),
            zero_division=0,
        )
        for index, label in enumerate(LABEL_ORDER):
            class_metric_rows.append(
                {
                    "model": model_name,
                    "class": label,
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1[index],
                    "support": int(support[index]),
                }
            )

        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predicted_labels,
            labels=range(len(LABEL_ORDER)),
            display_labels=LABEL_ORDER,
            normalize="true",
            values_format=".2f",
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(
            f"{model_name.replace('_', ' ').title()}\n"
            "Chromosome-Held-Out Predictions"
        )
        figure.tight_layout()
        figure.savefig(
            figures_dir / f"confusion_matrix_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    pd.DataFrame(class_metric_rows).to_csv(
        metrics_dir / "chromosome_holdout_per_class.csv",
        index=False,
    )


def fit_final_models(
    dataframe,
    feature_columns,
    seed,
    jobs,
    include_xgboost,
    output_path,
):
    """
    Fit models on all rows and save a prediction-ready bundle.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Complete feature table.
    feature_columns : list[str]
        Predictor columns used by each model.
    seed : int
        Random seed used by the models.
    jobs : int
        Processor threads used by XGBoost.
    include_xgboost : bool
        Whether to fit XGBoost.
    output_path : pathlib.Path
        Destination for the serialized model bundle.
    """
    means = dataframe[feature_columns].mean()
    standard_deviations = (
        dataframe[feature_columns].std().replace(0, 1)
    )
    predictors = (
        dataframe[feature_columns] - means
    ) / standard_deviations
    label_to_integer = {
        label: index for index, label in enumerate(LABEL_ORDER)
    }
    targets = dataframe["label"].map(label_to_integer).to_numpy()

    models = make_models(seed, jobs, include_xgboost)
    for name, model in models.items():
        print(f"fitting final {name} model on {len(dataframe)} rows")
        model.fit(predictors, targets)

    bundle = {
        "models": models,
        "feature_columns": feature_columns,
        "feature_means": means,
        "feature_standard_deviations": standard_deviations,
        "label_order": LABEL_ORDER,
        "window": len(
            [column for column in feature_columns if column.startswith("pos_")]
        )
        // 8,
    }
    joblib.dump(bundle, output_path)


def train_and_save(
    args,
    features_path,
    models_dir,
    metrics_dir,
    figures_dir,
):
    """
    Evaluate models, save metrics, and fit final models.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed pipeline settings.
    features_path : pathlib.Path
        Feature CSV used for modeling.
    models_dir : pathlib.Path
        Directory for serialized models.
    metrics_dir : pathlib.Path
        Directory for evaluation tables.
    figures_dir : pathlib.Path
        Directory for evaluation figures.
    """
    print("\n[loading features for modeling]")
    dataframe = pd.read_csv(features_path)
    split_column, feature_columns = get_modeling_columns(dataframe)
    print(
        f"loaded {len(dataframe)} rows with {len(feature_columns)} predictors"
    )

    folds = build_chromosome_folds(
        dataframe,
        split_column,
        args.folds,
    )
    fold_results, held_out_predictions = evaluate_models(
        dataframe=dataframe,
        split_column=split_column,
        feature_columns=feature_columns,
        folds=folds,
        seed=args.seed,
        jobs=args.jobs,
        include_xgboost=not args.skip_xgboost,
    )
    summary = (
        fold_results.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
        )
        .reset_index()
    )

    save_evaluation_outputs(
        fold_results=fold_results,
        summary=summary,
        held_out_predictions=held_out_predictions,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )
    print("\n[chromosome holdout summary]")
    print(summary.to_string(index=False))

    models_dir.mkdir(parents=True, exist_ok=True)
    fit_final_models(
        dataframe=dataframe,
        feature_columns=feature_columns,
        seed=args.seed,
        jobs=args.jobs,
        include_xgboost=not args.skip_xgboost,
        output_path=models_dir / "multiclass_models.joblib",
    )

    run_information = {
        "genome": str(args.genome),
        "annotation": str(args.annotation),
        "rows": len(dataframe),
        "predictors": len(feature_columns),
        "folds": folds,
        "seed": args.seed,
        "window": args.window,
        "hard_negative_fraction": args.hard_negative_fraction,
        "labels": LABEL_ORDER,
    }
    with (metrics_dir / "run_information.json").open("w") as handle:
        json.dump(run_information, handle, indent=2)


def main():
    args = parse_args()
    validate_args(args)

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    dataset_script = scripts_dir / "build_dataset.py"
    feature_script = scripts_dir / "build_features.py"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    models_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "splice_dataset.csv"
    features_path = data_dir / "features.csv"
    database_path = data_dir / "annotation.gffdb"

    if not args.skip_data:
        if not dataset_script.is_file():
            raise FileNotFoundError(f"Script not found: {dataset_script}")
        dataset_command = [
            sys.executable,
            str(dataset_script),
            "--genome",
            str(args.genome),
            "--annotation",
            str(args.annotation),
            "--database",
            str(database_path),
            "--output",
            str(dataset_path),
            "--window",
            str(args.window),
            "--seed",
            str(args.seed),
            "--hard-negative-fraction",
            str(args.hard_negative_fraction),
            "--transcript-feature",
            args.transcript_feature,
        ]
        for chromosome in args.exclude_chrom:
            dataset_command.extend(["--exclude-chrom", chromosome])
        if args.rebuild_db:
            dataset_command.append("--rebuild-db")
        run_command(dataset_command, "building dataset")
    elif not dataset_path.is_file():
        raise FileNotFoundError(
            f"--skip-data was used, but no dataset exists at {dataset_path}"
        )

    if not args.skip_features:
        if not feature_script.is_file():
            raise FileNotFoundError(f"Script not found: {feature_script}")
        feature_command = [
            sys.executable,
            str(feature_script),
            "--input",
            str(dataset_path),
            "--output",
            str(features_path),
        ]
        if args.deduplicate_sequences:
            feature_command.append("--deduplicate-sequences")
        run_command(feature_command, "building features")
    elif not features_path.is_file():
        raise FileNotFoundError(
            f"--skip-features was used, but no file exists at {features_path}"
        )

    if not args.skip_training:
        train_and_save(
            args,
            features_path,
            models_dir,
            metrics_dir,
            figures_dir,
        )

    print(f"\nPipeline complete. Outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
