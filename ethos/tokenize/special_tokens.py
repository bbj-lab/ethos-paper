from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ethos.tokenize.constants import TIME_DTYPE


class SpecialToken:
    SEPARATORS = {
        k: v / (60 * 24 * 365.25)
        for k, v in {
            "_5m-15m": 5,
            "_15m-1h": 15,
            "_1h-2h": 60,
            "_2h-6h": 2 * 60,
            "_6h-12h": 6 * 60,
            "_12h-1d": 12 * 60,
            "_1d-3d": 24 * 60,
            "_3d-1w": 3 * 24 * 60,
            "_1w-2w": 7 * 24 * 60,
            "_2w-1mt": 2 * 7 * 24 * 60,
            "_1mt-3mt": 30 * 24 * 60,
            "_3mt-6mt": 3 * 30 * 24 * 60,
            "_=6mt": 6 * 30 * 24 * 60,
        }.items()
    }
    SEPARATOR_NAMES = list(SEPARATORS.keys())
    SEPARATOR_SIZES = np.fromiter(SEPARATORS.values(), dtype=TIME_DTYPE)
    CLOCKS = {
        "_mdnt-1a": 0,
        "_1a-2a": 1,
        "_2a-3a": 2,
        "_3a-4a": 3,
        "_4a-5a": 4,
        "_5a-6a": 5,
        "_6a-7a": 6,
        "_7a-8a": 7,
        "_8a-9a": 8,
        "_9a-10a": 9,
        "_10a-11a": 10,
        "_11a-noon": 11,
        "_noon-1p": 12,
        "_1p-2p": 13,
        "_2p-3p": 14,
        "_3p-4p": 15,
        "_4p-5p": 16,
        "_5p-6p": 17,
        "_6p-7p": 18,
        "_7p-8p": 19,
        "_8p-9p": 20,
        "_9p-10p": 21,
        "_10p-11p": 22,
        "_11p-mdnt": 23,
    }
    CLOCK_NAMES = sorted(CLOCKS.keys(), key=CLOCKS.__getitem__)
    CLOCK_STARTS = sorted(CLOCKS.values())
    TIMELINE_END = "_TIMELINE_END"
    DEATH = "_DEATH"

    # e.g., used for defining age of a person
    YEARS = {"_<5": 5, **{f"_{i - 5}-{i}y": i for i in range(10, 101, 5)}}
    YEAR_NAMES = list(YEARS.keys()) + ["_>100"]
    YEAR_BINS = list(YEARS.values())

    DECILES = [f"_Q{i}" for i in range(1, 11)]

    ALL = [
        *SEPARATOR_NAMES,
        *CLOCK_NAMES,
        TIMELINE_END,
        DEATH,
        *YEAR_NAMES,
        *DECILES,
    ]

    @staticmethod
    def get_longest_separator() -> (str, float):
        return (
            SpecialToken.SEPARATOR_NAMES[-1],
            SpecialToken.SEPARATOR_SIZES[-1],
        )

    @staticmethod
    def age_to_year_token(age: float) -> str:
        i = np.digitize(age, SpecialToken.YEAR_BINS)
        return SpecialToken.YEAR_NAMES[i]

    @staticmethod
    def convert_to_deciles(
        values: pd.Series,
        bins: Optional[Iterable] = None,
        scheme: str = "quantiles",
        ret_bins: bool = False,
    ) -> pd.Series | tuple[pd.Series, np.ndarray[float]]:
        labels = SpecialToken.DECILES
        if bins is None:
            if scheme == "equidistant":
                bins = np.linspace(values.min(), values.max(), len(labels) + 1)
            elif scheme == "quantiles":
                bins = values.quantile(q=np.linspace(0, 1, len(labels) + 1))
            else:
                raise ValueError(
                    f"Got invalid `scheme` {scheme}, available: 'equidistant' and 'quantiles'."
                )
        bins = np.unique(bins)
        if np.isnan(bins[-1]):
            bins = bins[:-1]
        discrete_values = pd.cut(
            values,
            bins=bins,
            labels=labels[: len(bins) - 1],
            include_lowest=True,
        )
        # Don't make these values categorical, because later we might want to map nans into
        # something else, and doing it with categorical-valued series is cumbersome.
        discrete_values = discrete_values.astype(object)
        if ret_bins:
            return discrete_values, bins
        return discrete_values
