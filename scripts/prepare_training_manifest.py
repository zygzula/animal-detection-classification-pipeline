import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.dataframe_utils import filter_out_value, fetch_first_n
from src.utils.json_utils import load_json_list_to_dataframe

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_INPUT_PATH = BASE_DIR / "data" / "raw" / "metadata.json"
CATEGORIES_OUTPUT_PATH = BASE_DIR / "models" / "categories.json"
TRAINING_MANIFEST_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "training_manifest.json"

FILTER_COLUMN = "category_id"
FILTER_VALUE = 0
TOP_N = None


def load_categories_to_dataframe(json_path: Path) -> pd.DataFrame:
    return load_json_list_to_dataframe(json_path, "categories")


def load_images_to_dataframe(json_path: Path) -> pd.DataFrame:
    return load_json_list_to_dataframe(json_path, "images")


def load_annotations_to_dataframe(json_path: Path) -> pd.DataFrame:
    return load_json_list_to_dataframe(json_path, "annotations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Animal classifier prediction")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output: lists of columns for the processed dataframes"
    )

    args = parser.parse_args()
    verbose = args.verbose

    if not DATA_INPUT_PATH.exists():
        raise FileNotFoundError(f"JSON file was not found: {DATA_INPUT_PATH.resolve()}")

    images_metadata_df = load_images_to_dataframe(DATA_INPUT_PATH)
    annotations_metadata_df = load_annotations_to_dataframe(DATA_INPUT_PATH)

    print("\nData loaded successfully.")
    print(f"Shape: [images: {images_metadata_df.shape}] [annotations: {annotations_metadata_df.shape}]")

    if verbose:
        print(f"Images columns: [{images_metadata_df.columns.tolist()}]")
        print(f"Annotations columns: [{annotations_metadata_df.columns.tolist()}]")

    filtered_df = filter_out_value(
        annotations_metadata_df,
        filtered_column=FILTER_COLUMN,
        filtered_value=FILTER_VALUE
    )

    cropped_df = fetch_first_n(df=filtered_df, n=TOP_N)

    merged_df = cropped_df.merge(
        images_metadata_df,
        left_on="image_id",
        right_on="id",
        how="left"
    )
    if merged_df["image_id"].isna().any():
        raise ValueError("Some annotations could not be matched to image metadata.")

    unique_df = merged_df[~merged_df["image_id"].duplicated(keep=False)]
    metadata_output_list = unique_df.to_dict(orient="records")

    TRAINING_MANIFEST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_MANIFEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_output_list, f, ensure_ascii=False, indent=2)

    categories_df = load_categories_to_dataframe(DATA_INPUT_PATH)
    categories_output_list = categories_df.to_dict(orient="records")

    CATEGORIES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORIES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(categories_output_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
