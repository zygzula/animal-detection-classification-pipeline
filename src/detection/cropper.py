from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from megadetector.detection.run_detector_batch import load_and_run_detector_batch

MD_ANIMAL_CATEGORY = "1"


def _is_animal_detection(detection: Dict[str, Any]) -> bool:
    return str(detection.get("category")) == MD_ANIMAL_CATEGORY


def _filter_animal_detections(
        detections: List[Dict[str, Any]],
        confidence_threshold: float,  # used as LOW threshold
) -> List[Dict[str, Any]]:
    multi_detection_threshold = max(0.6, confidence_threshold)

    animal_detections = [
        detection
        for detection in detections
        if _is_animal_detection(detection)
    ]

    if not animal_detections:
        return []

    animal_detections.sort(key=lambda d: d.get("conf", 0.0), reverse=True)

    if len(animal_detections) == 1:
        if animal_detections[0].get("conf", 0.0) >= confidence_threshold:
            return animal_detections

        return []

    else:
        filtered = [
            d for d in animal_detections
            if d.get("conf", 0.0) >= multi_detection_threshold
        ]

        return filtered


def _save_detection_crops(
        image_path: Path,
        detections: List[Dict[str, Any]],
        category_name: str,
        output_dir: Path,
        padding_ratio: float = 0.05,
        min_crop_width: int = 32,
        min_crop_height: int = 32,
) -> int:
    saved_count = 0
    stem = image_path.stem
    suffix = image_path.suffix

    with Image.open(image_path) as image:
        image_width, image_height = image.size

        for index, detection in enumerate(detections):
            x, y, w, h = detection["bbox"]

            left = x * image_width
            top = y * image_height
            right = (x + w) * image_width
            bottom = (y + h) * image_height

            left, top, right, bottom = apply_padding(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
            )

            crop_width = right - left
            crop_height = bottom - top

            if crop_width < min_crop_width or crop_height < min_crop_height:
                continue

            crop = image.crop((left, top, right, bottom))

            output_path = output_dir / category_name / f"{stem}_{index}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)

            saved_count += 1

    return saved_count


def apply_padding(
        left: float,
        top: float,
        right: float,
        bottom: float,
        image_width: int,
        image_height: int,
        padding_ratio: float,
) -> Tuple[int, int, int, int]:
    box_width = right - left
    box_height = bottom - top

    padding_x = box_width * padding_ratio
    padding_y = box_height * padding_ratio

    padded_left = max(0, int(left - padding_x))
    padded_top = max(0, int(top - padding_y))
    padded_right = min(image_width, int(right + padding_x))
    padded_bottom = min(image_height, int(bottom + padding_y))

    return padded_left, padded_top, padded_right, padded_bottom


def run_detection_and_crop(
        batch_items: List[Dict[str, Any]],
        categories_dict: Dict[Any, str],
        model_path: Path,
        output_dir: Path,
        confidence_threshold: float = 0.2,
        keep_only_best_animal_detection: bool = True,
        padding_ratio: float = 0.05,
        min_crop_width: int = 32,
        min_crop_height: int = 32,
) -> int:
    image_paths = [item["local_path"] for item in batch_items]
    items_by_path = {item["local_path"]: item for item in batch_items}

    results = load_and_run_detector_batch(
        model_file=str(model_path),
        image_file_names=image_paths,
        confidence_threshold=confidence_threshold,
        quiet=True,
    )

    total_saved = 0
    for image_result in results:
        file_path_str = image_result["file"]
        detections = image_result.get("detections", [])

        item = items_by_path[file_path_str]
        category_id = item["category_id"]
        category_name = categories_dict.get(category_id)

        if category_name is None:
            print(f"Unknown category_id={category_id} for {file_path_str}")
            continue

        animal_detections = _filter_animal_detections(
            detections=detections,
            confidence_threshold=confidence_threshold,
        )

        if not animal_detections:
            continue

        if keep_only_best_animal_detection:
            animal_detections = animal_detections[:1]

        total_saved += _save_detection_crops(
            image_path=Path(file_path_str),
            detections=animal_detections,
            category_name=category_name,
            output_dir=output_dir,
            padding_ratio=padding_ratio,
            min_crop_width=min_crop_width,
            min_crop_height=min_crop_height,
        )

    return total_saved
