from typing import Any

import pandas as pd

DEFAULT_N = 100


def filter_out_value(
        df: pd.DataFrame,
        filtered_column: str,
        filtered_value: Any
) -> pd.DataFrame:
    if filtered_column not in df.columns:
        raise KeyError(f"Column '{filtered_column}' was not found.")

    return df[df[filtered_column] != filtered_value]


def fetch_first_n(df: pd.DataFrame, n: int | None = DEFAULT_N) -> pd.DataFrame:
    if n is None:
        return df

    if n <= 0:
        raise ValueError(f"n={n} has to be greater than 0.")

    return df.head(n)
