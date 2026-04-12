import argparse
import json
import math
from typing import List, Dict, Any

import requests
import torch

import src.config as config
from src.data.download import prepare_batch, delete_temporary_images
from src.detection.cropper import run_detection_and_crop
from src.utils.logging_utils import log_message, clear_log_file, log_and_print_message


def create_categories_dict(categories_json_list: List[Dict[str, Any]]) -> Dict[Any, str]:
    return {item["id"]: item["name"] for item in categories_json_list}


def main() -> None:
    parser = argparse.ArgumentParser(description="Animal classifier prediction")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output: lists of columns for the processed dataframes"
    )

    args = parser.parse_args()
    verbose = args.verbose

    torch.set_num_threads(8)
    clear_log_file(config.LOGS_FILENAME)

    with config.TRAINING_MANIFEST_PATH.open("r", encoding="utf-8") as file_obj:
        training_manifest_json = json.load(file_obj)

    with config.CATEGORIES_PATH.open("r", encoding="utf-8") as file_obj:
        categories_json = json.load(file_obj)

    categories_dict = create_categories_dict(categories_json)

    if verbose:
        print('Categories dictionary:', categories_dict)

    total_images = len(training_manifest_json)
    total_batches = math.ceil(total_images / config.BATCH_SIZE)

    total_downloaded = 0
    total_crops_saved = 0

    log_and_print_message(
        f"Estimated number of batches: {total_batches} "
        f"(images in total: {total_images}; batch size: {config.BATCH_SIZE})",
        config.LOGS_FILENAME
    )

    with requests.Session() as session:
        for batch_number in range(total_batches):
            log_and_print_message(f"Processing batch {batch_number + 1}/{total_batches}", config.LOGS_FILENAME)

            batch_items = prepare_batch(
                manifest_items=training_manifest_json,
                download_dir=config.IMAGE_INPUT_DIR,
                batch_number=batch_number,
                batch_size=config.BATCH_SIZE,
                session=session,
            )

            downloaded_count = len(batch_items)
            total_downloaded += downloaded_count

            download_percentage_log = f"Successfully prepared {downloaded_count}/{config.BATCH_SIZE} images"
            log_message(f"[{batch_number + 1}/{total_batches}] {download_percentage_log}", config.LOGS_FILENAME)
            print(download_percentage_log)

            if downloaded_count == 0:
                continue

            crops_saved = run_detection_and_crop(
                batch_items=batch_items,
                categories_dict=categories_dict,
                model_path=config.MODEL_PATH,
                output_dir=config.IMAGE_OUTPUT_DIR,
                confidence_threshold=config.DETECTION_CONFIDENCE_THRESHOLD,
                keep_only_best_animal_detection=True,
            )

            total_crops_saved += crops_saved
            delete_temporary_images(config.IMAGE_INPUT_DIR)

    log_and_print_message("Processing finished.", config.LOGS_FILENAME)
    log_and_print_message(f"Images prepared: {total_downloaded}", config.LOGS_FILENAME)
    log_and_print_message(f"Crops saved: {total_crops_saved}", config.LOGS_FILENAME)


if __name__ == "__main__":
    main()
