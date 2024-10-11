import time
from typing import Optional

import numpy as np
from click import Choice, command, option, argument

from ethos.constants import PROJECT_DATA
from ethos.tokenize import Vocabulary
from ethos.tokenize.constants import Dataset, DataFold, DataProp
from ethos.tokenize.tokenization_steps import (
    concatenate_timelines,
    count_clock_tokens,
    dump_timelines,
    estimate_true_sep_time,
    get_context_data,
    inject_separators,
    load_and_process_data,
    merge_timeline_chunks,
    normalize_mimic_times,
)
from ethos.utils import convert_years, convert_seconds, get_logger

logger = get_logger()

DEFAULT_OUT_DIR = "tokenized_datasets"


@command()
@argument(
    "dataset",
    default="mimic",
    required=True,
    type=Choice([d.value for d in Dataset]),
)
@argument(
    "fold",
    default="test",
    required=True,
    type=Choice([f.value for f in DataFold]),
)
@option(
    "-v", "--vocab_path", default=None, help="Path to the existing vocabulary."
)
@option(
    "-n",
    "--nrows",
    default=None,
    type=int,
    help="Number of rows to load, `None` for all.",
)
@option(
    "-o",
    "--out",
    default=DEFAULT_OUT_DIR,
    help="Path of the output directory in PROJECT_DATA.",
    show_default=True,
)
@option(
    "-j",
    "--n_jobs",
    default=1,
    help="Number of processes used to inject separators.",
)
@option(
    "-p",
    "--plots",
    default=True,
    help="Boolean call for plots.",
)
@option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for the order of patient in the data.",
)
def tokenize_(
    dataset: str,
    fold: str,
    nrows: Optional[int],
    vocab_path: Optional[str],
    out: str,
    n_jobs: int,
    seed: int,
    plots: bool,
):
    """Tokenize one of the folds of the MIMIC dataset. The dataset files are
    expected to be in the PROJECT_DATA directory. See README.md for more information.

    The name is weird due to the Click issue: https://github.com/pallets/click/issues/2615
    """
    dataset_prop = DataProp.create(dataset, fold)
    if nrows is None:
        # use parquet files for the entire dataset as it's way faster
        kwargs = dict(use_parquet=True)
    else:
        kwargs = dict(nrows=nrows, use_parquet=False)

    output_dir = PROJECT_DATA / out
    logger.info(
        f"Generating timelines for {fold} of the {dataset.upper()} dataset"
    )
    start_time = time.time()

    vocab = Vocabulary(vocab_path)
    if vocab_path is not None:
        logger.info(f"Loaded an exising vocabulary of size: {len(vocab)}")

    logger.info(f"Getting context data...")
    context_data, age_reference = get_context_data(
        dataset_prop, vocab, **kwargs
    )

    logger.info(f"Processing time data...")
    patient_timeline_chunks = load_and_process_data(
        dataset_prop, vocab, **kwargs
    )
    logger.info(f"Got timelines of {len(patient_timeline_chunks):,} patients")
    logger.info(f"Vocabulary size: {len(vocab):,}")

    logger.info(
        f"Merging timeline chunks retrieved from different data subsets..."
    )
    patient_timelines = merge_timeline_chunks(patient_timeline_chunks)

    if dataset_prop.name == Dataset.MIMIC:
        patient_timelines = normalize_mimic_times(
            patient_timelines, dataset_prop, **kwargs
        )

    # shuffle the patients, so we are sure that the patients are not in any biased order,
    # first we sort them, so that the order is deterministic
    logger.info(f"Shuffling patients...")
    patient_ids = sorted(patient_timelines.keys())
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    patient_timelines = {
        patient_id: patient_timelines[patient_id] for patient_id in patient_ids
    }

    logger.info(f"Injecting separators into timelines...")
    postprocess_start_time = time.time()
    patient_timelines = inject_separators(
        patient_timelines, vocab, n_jobs=n_jobs
    )
    separator_time = time.time() - postprocess_start_time

    log_statistics_about_timelines(patient_timelines)
    if plots:
        plot_n_timelines(patient_timelines, vocab)

    logger.info(f"Concatenating all timelines...")
    times, tokens, patient_idx_to_id = concatenate_timelines(patient_timelines)

    logger.info("Separators statistics:")
    estimate_true_sep_time(times, tokens, vocab)
    log_separator_statistics(vocab)

    logger.info("Counts of clock tokens:")
    d = count_clock_tokens(tokens, vocab)
    log_clock_token_counts(d)
    if plots:
        plot_clock_histogram(d)

    logger.info("Logging top 10 tokens near midnight...")
    log_top_tokens_with_clocks(times, tokens, vocab)

    output_dir.mkdir(parents=True, exist_ok=True)
    timelines_path = (
        output_dir
        / f"{dataset}_{fold}_timelines_p{len(patient_idx_to_id)}.hdf5"
    )
    logger.info(f"Dumping the timelines to '{timelines_path}'")
    dump_timelines(
        timelines_path,
        times,
        tokens,
        patient_idx_to_id,
        context_data,
        age_reference,
    )

    if vocab_path is None:
        vocab_path = output_dir / f"{dataset}_vocab_t{len(vocab)}.pkl"
        logger.info(f"Dumping the vocabulary to '{vocab_path}'")
        vocab.to_pickle(vocab_path)

    logger.info(
        "Timelines generated in {}, separator injection took {}".format(
            convert_seconds(time.time() - start_time),
            convert_seconds(separator_time),
        )
    )


def log_statistics_about_timelines(patient_timelines):
    stats = {"MIN": 0, "Q1": 0.25, "MEDIAN": 0.5, "Q3": 0.75, "MAX": 1}

    patient_timelines_lengths = np.array(
        [len(v[0]) for v in patient_timelines.values()]
    )
    logger.info(
        "Total number of tokens: {:,} - mean timeline length: {:.2f} (+/- {:.2f})".format(
            np.sum(patient_timelines_lengths),
            np.mean(patient_timelines_lengths),
            np.std(patient_timelines_lengths),
        )
    )
    quartiles = np.quantile(patient_timelines_lengths, list(stats.values()))
    logger.info(
        "Timeline length quartiles: "
        + ", ".join(
            [
                "{}={:.0f}".format(q_label, q)
                for q_label, q in zip(stats.keys(), quartiles)
            ]
        )
    )


def log_separator_statistics(vocab: Vocabulary):
    logger.info(
        f"  Separator contribution: {vocab.meta['separator_contrib']:.2%}"
    )
    logger.info("  Separator estimates:")
    separator_estimates = vocab.meta["separator_estimates"]
    for separator in separator_estimates["mean"].keys():
        unit = "".join(c for c in separator.split("-")[0] if c.isalpha())
        min_ = convert_years(separator_estimates["min"][separator], unit)
        median = convert_years(separator_estimates["median"][separator], unit)
        max_ = convert_years(separator_estimates["max"][separator], unit)
        count = separator_estimates["count"][separator]
        logger.info(
            f"  {separator:<8} (unit='{unit}'): min={min_:.3f}, median={median:.3f},"
            f" max={max_:.3f} (count={count:,})"
        )


def log_clock_token_counts(d: dict):
    total_clock_tokens = sum(d.values())
    for k, v in d.items():
        logger.info(
            f"  {k:<9} total: {v:>7}, faction: {v/total_clock_tokens:.4f}"
        )


def plot_n_timelines(patient_timelines, vocab, n_tls=100, t_trunc=1000):
    import plotly.express as px
    from ethos.tokenize.special_tokens import SpecialToken

    cat_lookup = (
        {v: 1 for v in vocab.stoi.values()}
        | {vocab.stoi[s]: 2 for s in SpecialToken.SEPARATOR_NAMES}
        | {vocab.stoi[s]: 3 for s in SpecialToken.CLOCK_NAMES}
        | {-1: 0}
    )
    fix_tl = lambda tl: list(tl)[:t_trunc] + [-1] * (t_trunc - len(tl))
    tls = [
        fix_tl(patient_timelines[k][1])
        for k in list(patient_timelines.keys())[:n_tls]
    ]
    tls_cats = [[cat_lookup[t] for t in x] for x in tls]
    fig = px.imshow(
        np.array(tls_cats, dtype=np.uint8),
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.show()


def plot_clock_histogram(d: dict):
    import pandas as pd
    import plotly.express as px
    from ethos.tokenize.special_tokens import SpecialToken

    df = (
        pd.DataFrame.from_dict(d, orient="index", columns=["count"])
        .rename_axis(index="clock")
        .reset_index()
    )
    fig = px.bar(df, x="clock", y="count")
    fig.update_xaxes(
        categoryorder="array", categoryarray=SpecialToken.CLOCK_NAMES
    )
    fig.show()


def log_top_tokens_with_clocks(times, tokens, vocab, clocks_ls=(0, 23), k=10):
    from ethos.tokenize.separators import tm2hr
    from ethos.tokenize.special_tokens import SpecialToken

    tokens_clk = [
        vtk
        for tm, tk in zip(times, tokens)
        if tm2hr(tm) in clocks_ls
        and (vtk := vocab.itos[tk]) not in SpecialToken.ALL
    ]
    get_counts = lambda v: dict(zip(*np.unique(v, return_counts=True)))
    tokens_clk_ct = get_counts(tokens_clk)
    for k, v in sorted(
        tokens_clk_ct.items(), key=lambda x: x[1], reverse=True
    )[:k]:
        logger.info("{:>7}: {}".format(v, k))


if __name__ == "__main__":
    tokenize_()
