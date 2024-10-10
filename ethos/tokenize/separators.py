from datetime import datetime, timezone
from itertools import zip_longest, chain
from typing import Iterable

import numpy as np

from . import Vocabulary
from .constants import TOKEN_DTYPE, TIME_DTYPE, SECONDS_IN_YEAR
from .special_tokens import SpecialToken


def tm2hr(tm: float) -> int:
    """convert curious floating year representation into an hour 0-23"""
    return int(
        datetime.fromtimestamp(tm * SECONDS_IN_YEAR, tz=timezone.utc).strftime(
            "%H"
        )
    )


class SeparatorInjector:
    def __init__(self, vocab: Vocabulary):
        sep_time_expectations = (
            SpecialToken.SEPARATOR_SIZES[1:]
            - SpecialToken.SEPARATOR_SIZES[:-1]
        ) / 2
        sep_time_expectations = [
            *sep_time_expectations,
            SpecialToken.SEPARATOR_SIZES[-1],
        ]
        name_to_time = dict(
            zip(SpecialToken.SEPARATOR_NAMES, sep_time_expectations)
        )
        # we start from 1 because 0 is the bin that times fall below the shortest separator
        self._sep_bin_to_token = dict(
            zip(
                range(1, len(SpecialToken.SEPARATOR_NAMES) + 1),
                SpecialToken.SEPARATOR_NAMES,
            )
        )
        self._clock_bin_to_token = dict(
            zip(
                range(1, len(SpecialToken.CLOCK_NAMES) + 1),
                SpecialToken.CLOCK_NAMES,
            )
        )
        self._sep_token_to_time = {
            vocab.encode(sep_name): name_to_time[sep_name]
            for i, sep_name in enumerate(SpecialToken.SEPARATOR_NAMES, 1)
        }
        self._timeline_end_token = vocab.encode(SpecialToken.TIMELINE_END)
        self._clk2int = {c: vocab.stoi[c] for c in SpecialToken.CLOCK_NAMES}

    def __call__(
        self,
        sorted_times: np.ndarray[float],
        sorted_event_tokens: Iterable[TOKEN_DTYPE],
    ) -> (np.ndarray[TIME_DTYPE], np.ndarray[TOKEN_DTYPE]):
        """The input must be sorted, adds also TIMELINE_END token."""
        deltas = sorted_times[1:] - sorted_times[:-1]
        # longest separator occurs multiple times if possible
        longest_sep, longest_sep_dur = SpecialToken.get_longest_separator()
        binned_deltas = np.digitize(deltas, SpecialToken.SEPARATOR_SIZES)
        separator_tokens = (
            (
                [token] * round(deltas[i] / longest_sep_dur)
                if token == longest_sep
                else token
            )
            for i, token in enumerate(binned_deltas)
        )
        binned_times = np.digitize(
            [tm2hr(t) for t in sorted_times], SpecialToken.CLOCK_STARTS
        )
        clock_tokens = (self._clock_bin_to_token[t] for t in binned_times)

        clocked = False
        tokens = []

        for clock_token, event_token, sep_token in zip_longest(
            clock_tokens, sorted_event_tokens, separator_tokens
        ):
            if not clocked:
                tokens.append(
                    self._conv_token_to_uint(
                        clock_token, self._timeline_end_token
                    )
                )
                clocked = True
            tokens.append(
                self._conv_token_to_uint(event_token, self._timeline_end_token)
            )
            if sep_token != 0:
                if isinstance(sep_token, list):
                    tokens += [
                        self._conv_token_to_uint(s, self._timeline_end_token)
                        for s in sep_token
                    ]
                else:
                    tokens.append(
                        self._conv_token_to_uint(
                            sep_token, self._timeline_end_token
                        )
                    )
                clocked = False

        tokens = np.array(tokens, dtype=TOKEN_DTYPE)

        times_iter = JanusIterator(sorted_times)
        times = np.fromiter(
            (
                self._conv_token_to_time(
                    token, times_iter, self._timeline_end_token
                )
                for token in tokens
            ),
            dtype=TIME_DTYPE,
            count=len(tokens),
        )
        return times, tokens

    def _conv_token_to_uint(
        self, token: TOKEN_DTYPE, timeline_end_token: TOKEN_DTYPE
    ) -> TOKEN_DTYPE:
        if token in self._sep_bin_to_token:
            return token - 1
        elif token in self._clk2int:
            return self._clk2int[token]
        elif token is None:
            return timeline_end_token
        return token

    def _conv_token_to_time(
        self, token: TOKEN_DTYPE, times_iter, timeline_end_token: TOKEN_DTYPE
    ) -> TIME_DTYPE:
        """For regular tokens, just returns their time, while separators are estimated."""
        if token in self._sep_token_to_time:
            return times_iter.prev + self._sep_token_to_time[token]
        elif token in self._clk2int.values():
            return times_iter.next
        elif token == timeline_end_token:
            return times_iter.prev
        return next(times_iter)


class JanusIterator:
    """looks forward and backward"""
    def __init__(self, iterable):
        self.iterator = chain(iter(iterable), [None])
        self.prev = None
        self.next = next(self.iterator)

    def __iter__(self):
        return self

    def __next__(self):
        self.prev = self.next
        self.next = next(self.iterator)
        return self.prev
