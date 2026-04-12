import json
from pathlib import Path

import pandas as pd


def load_json_list_to_dataframe(json_path: Path | str, list_key: str | None = None) -> pd.DataFrame:
    if isinstance(json_path, str):
        json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if list_key is not None:
            raise ValueError(f"The object is already a list. Contradictory list_key={list_key} parameter.")

        return pd.json_normalize(data)

    if isinstance(data, dict):
        if list_key is not None:
            json_list = data.get(list_key)
            if json_list is None:
                raise ValueError(f"The object does not contain the key={list_key}.")
            elif not isinstance(json_list, list):
                raise ValueError(f"The value under key={list_key} is not a list.")

            return pd.json_normalize(json_list)
        else:
            for value in data.values():
                if isinstance(value, list):
                    return pd.json_normalize(value)

            raise ValueError("The root dictionary object does not contain any list.")

    raise ValueError("Incompatible JSON structure.")
