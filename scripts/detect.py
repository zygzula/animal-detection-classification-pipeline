import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

import src.config as config
from src.detection.cropper import MD_ANIMAL_CATEGORY, apply_padding


def get_class_names_from_dir(dataset_path: Path) -> list[str]:
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Classes directory not found: {dataset_path}")

    return sorted(
        item.name
        for item in dataset_path.iterdir()
        if item.is_dir()
    )


def find_image_files(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    return sorted(
        path
        for path in dir_path.iterdir()
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def load_and_pad_crop(
        crop: Image.Image,
        target_size: tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    crop_array = np.asarray(crop, dtype=np.float32)
    crop_tensor = tf.convert_to_tensor(crop_array)

    crop_resized = tf.image.resize_with_pad(
        crop_tensor,
        target_size[0],
        target_size[1],
    )

    return crop_resized.numpy()


def prepare_crop_for_model(
        crop: Image.Image,
        target_size: tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    crop_array = load_and_pad_crop(crop, target_size)
    crop_array = preprocess_input(crop_array)
    return np.expand_dims(crop_array, axis=0)


def classify_crop(
        classifier_model: tf.keras.Model,
        crop: Image.Image,
        class_names: list[str],
        img_size: tuple[int, int] = config.IMG_SIZE,
) -> dict[str, Any]:
    crop_array = prepare_crop_for_model(crop, img_size)

    predictions = classifier_model.predict(crop_array, verbose=0)
    probs = predictions[0]

    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    confidence = float(probs[predicted_index])

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
    }


def build_detection_record(
        detection: dict[str, Any],
        image: Image.Image,
        classifier_model: tf.keras.Model,
        class_names: list[str],
        classifier_img_size: tuple[int, int],
        classifier_crop_padding_ratio: float,
        visualization_padding_ratio: float,
        min_crop_width: int,
        min_crop_height: int,
) -> dict[str, Any] | None:
    image_width, image_height = image.size

    detector_category_id = str(detection.get("category"))
    detector_label = config.DETECTOR_CATEGORY_MAP.get(
        detector_category_id,
        f"unknown_{detector_category_id}",
    )
    detector_confidence = float(detection.get("conf", 0.0))

    bbox = detection.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None

    x, y, bbox_width, bbox_height = bbox

    original_left = x * image_width
    original_top = y * image_height
    original_right = (x + bbox_width) * image_width
    original_bottom = (y + bbox_height) * image_height

    classifier_left, classifier_top, classifier_right, classifier_bottom = apply_padding(
        left=original_left,
        top=original_top,
        right=original_right,
        bottom=original_bottom,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=classifier_crop_padding_ratio,
    )

    visualization_left, visualization_top, visualization_right, visualization_bottom = apply_padding(
        left=original_left,
        top=original_top,
        right=original_right,
        bottom=original_bottom,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=visualization_padding_ratio,
    )

    crop_width = classifier_right - classifier_left
    crop_height = classifier_bottom - classifier_top

    if crop_width < min_crop_width or crop_height < min_crop_height:
        return None

    detection_record: dict[str, Any] = {
        "detector_category_id": detector_category_id,
        "detector_label": detector_label,
        "detector_confidence": detector_confidence,
        "classifier_bbox_xyxy": {
            "left": int(classifier_left),
            "top": int(classifier_top),
            "right": int(classifier_right),
            "bottom": int(classifier_bottom),
        },
        "visualization_bbox_xyxy": {
            "left": int(visualization_left),
            "top": int(visualization_top),
            "right": int(visualization_right),
            "bottom": int(visualization_bottom),
        },
        "original_bbox_xyxy": {
            "left": int(original_left),
            "top": int(original_top),
            "right": int(original_right),
            "bottom": int(original_bottom),
        },
    }

    if detector_category_id == MD_ANIMAL_CATEGORY:
        crop = image.crop((classifier_left, classifier_top, classifier_right, classifier_bottom))
        classification = classify_crop(
            classifier_model=classifier_model,
            crop=crop,
            class_names=class_names,
            img_size=classifier_img_size,
        )

        detection_record["classification"] = classification
        detection_record["display_label"] = (
            f"{classification['predicted_label']} "
            f"({classification['confidence']:.2%})"
        )
    else:
        detection_record["classification"] = None
        detection_record["display_label"] = (
            f"{detector_label} ({detector_confidence:.2%})"
        )

    return detection_record


def process_detector_result(
        image_path: Path,
        detections: list[dict[str, Any]],
        classifier_model: tf.keras.Model,
        class_names: list[str],
        classifier_img_size: tuple[int, int] = config.IMG_SIZE,
        classifier_crop_padding_ratio: float = config.CROP_PADDING,
        visualization_padding_ratio: float = 0.0,
        min_crop_width: int = config.MIN_CROP_WIDTH,
        min_crop_height: int = config.MIN_CROP_HEIGHT,
) -> dict[str, Any]:
    annotated_detections: list[dict[str, Any]] = []

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        for detection in detections:
            detection_record = build_detection_record(
                detection=detection,
                image=image,
                classifier_model=classifier_model,
                class_names=class_names,
                classifier_img_size=classifier_img_size,
                classifier_crop_padding_ratio=classifier_crop_padding_ratio,
                visualization_padding_ratio=visualization_padding_ratio,
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
            )
            if detection_record is not None:
                annotated_detections.append(detection_record)

    return {
        "file_path": str(image_path),
        "file_name": image_path.name,
        "detections": annotated_detections,
    }


def detect_and_classify_single_image(
        image_path: Path,
        detector_model_path: Path,
        classifier_model: tf.keras.Model,
        class_names: list[str],
        detection_confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
        classifier_img_size: tuple[int, int] = config.IMG_SIZE,
        classifier_crop_padding_ratio: float = config.CROP_PADDING,
        visualization_padding_ratio: float = 0.0,
        min_crop_width: int = config.MIN_CROP_WIDTH,
        min_crop_height: int = config.MIN_CROP_HEIGHT,
) -> dict[str, Any]:
    results = load_and_run_detector_batch(
        model_file=str(detector_model_path),
        image_file_names=[str(image_path)],
        confidence_threshold=detection_confidence_threshold,
        quiet=True,
    )

    if not results:
        return {
            "file_path": str(image_path),
            "file_name": image_path.name,
            "detections": [],
        }

    image_result = results[0]
    detections = image_result.get("detections", [])

    return process_detector_result(
        image_path=image_path,
        detections=detections,
        classifier_model=classifier_model,
        class_names=class_names,
        classifier_img_size=classifier_img_size,
        classifier_crop_padding_ratio=classifier_crop_padding_ratio,
        visualization_padding_ratio=visualization_padding_ratio,
        min_crop_width=min_crop_width,
        min_crop_height=min_crop_height,
    )


def detect_and_classify_directory(
        dir_path: Path,
        detector_model_path: Path,
        classifier_model: tf.keras.Model,
        class_names: list[str],
        detection_confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
        classifier_img_size: tuple[int, int] = config.IMG_SIZE,
        classifier_crop_padding_ratio: float = config.CROP_PADDING,
        visualization_padding_ratio: float = 0.0,
        min_crop_width: int = config.MIN_CROP_WIDTH,
        min_crop_height: int = config.MIN_CROP_HEIGHT,
) -> list[dict[str, Any]]:
    image_paths = find_image_files(dir_path)
    if not image_paths:
        return []

    raw_results = load_and_run_detector_batch(
        model_file=str(detector_model_path),
        image_file_names=[str(path) for path in image_paths],
        confidence_threshold=detection_confidence_threshold,
        quiet=True,
    )

    detections_by_path = {
        Path(item["file"]): item.get("detections", [])
        for item in raw_results
    }

    results: list[dict[str, Any]] = []
    for image_path in image_paths:
        result = process_detector_result(
            image_path=image_path,
            detections=detections_by_path.get(image_path, []),
            classifier_model=classifier_model,
            class_names=class_names,
            classifier_img_size=classifier_img_size,
            classifier_crop_padding_ratio=classifier_crop_padding_ratio,
            visualization_padding_ratio=visualization_padding_ratio,
            min_crop_width=min_crop_width,
            min_crop_height=min_crop_height,
        )
        results.append(result)

    return results


def draw_detections_on_image(
        image_path: Path,
        result: dict[str, Any],
        show: bool = False,
        save_vis_dir: Path | None = None,
        figure_width: int = 10,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        fig, ax = plt.subplots(figsize=(figure_width, figure_width))
        ax.imshow(image)
        ax.axis("off")

        for detection in result["detections"]:
            bbox = detection["visualization_bbox_xyxy"]
            left = bbox["left"]
            top = bbox["top"]
            right = bbox["right"]
            bottom = bbox["bottom"]

            width = right - left
            height = bottom - top

            detector_label = detection["detector_label"]

            if detector_label == "animal":
                edge_color = "lime"
            elif detector_label == "person":
                edge_color = "orange"
            elif detector_label == "vehicle":
                edge_color = "cyan"
            else:
                edge_color = "red"

            rect = patches.Rectangle(
                (left, top),
                width,
                height,
                linewidth=2,
                edgecolor=edge_color,
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.text(
                left,
                max(0, top - 5),
                detection["display_label"],
                fontsize=10,
                color="white",
                bbox=dict(facecolor=edge_color, alpha=0.8, pad=2),
            )

        if save_vis_dir is not None:
            save_vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_vis_dir / f"{image_path.stem}_detected.png"
            plt.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


def save_results_to_json(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)


def print_single_result(result: dict[str, Any], verbose: bool = False) -> None:
    print(f"File: {result['file_name']}")
    print(f"Detections: {len(result['detections'])}")

    for idx, detection in enumerate(result["detections"], start=1):
        print(f"  [{idx}] {detection['display_label']}")

        if verbose:
            print(f"       detector_label: {detection['detector_label']}")
            print(f"       detector_confidence: {detection['detector_confidence']:.4f}")

            classification = detection.get("classification")
            if classification is not None:
                print("       top classifier probabilities:")
                sorted_probs = sorted(
                    classification["probabilities"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                for class_name, prob in sorted_probs:
                    print(f"         - {class_name}: {prob:.4f}")

    print()


def print_directory_summary(results: list[dict[str, Any]]) -> None:
    print(f"Processed {len(results)} image(s).")

    total_detections = sum(len(result["detections"]) for result in results)
    print(f"Total detections: {total_detections}")

    label_counts: dict[str, int] = {}

    for result in results:
        for detection in result["detections"]:
            classification = detection.get("classification")
            if classification is not None:
                label = classification["predicted_label"]
            else:
                label = detection["detector_label"]

            label_counts[label] = label_counts.get(label, 0) + 1

    if label_counts:
        print("Label summary:")
        for label, count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {label}: {count}")


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together.")

    if args.show and args.dir:
        parser.error("--show is intended for single-file mode only.")

    if not 0 <= args.detection_threshold <= 1:
        parser.error("--detection-threshold must be between 0 and 1.")

    if args.classifier_crop_padding_ratio < 0:
        parser.error("--classifier-crop-padding-ratio must be >= 0.")

    if args.visualization_padding_ratio < 0:
        parser.error("--visualization-padding-ratio must be >= 0.")

    if args.min_crop_width <= 0 or args.min_crop_height <= 0:
        parser.error("--min-crop-width and --min-crop-height must be > 0.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MegaDetector, classify animal detections into species, and visualize results."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a single image file")
    group.add_argument("-d", "--dir", type=str, help="Path to a directory of images")

    parser.add_argument(
        "--detector-model",
        type=str,
        default=config.DETECTOR_MODEL_PATH,
        help=f"Path to MegaDetector model file (default: {config.DETECTOR_MODEL_PATH})",
    )

    parser.add_argument(
        "--classifier-model",
        type=str,
        default=config.CLASSIFIER_MODEL_BEST_PATH,
        help=f"Path to trained classifier model (default: {config.CLASSIFIER_MODEL_BEST_PATH})",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=config.IMAGE_OUTPUT_DIR,
        help=f"Path to class folders used to derive class names (default: {config.IMAGE_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=config.IMG_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help=f"Classifier input image size (default: {config.IMG_SIZE[0]} {config.IMG_SIZE[1]})",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=config.DETECTION_CONFIDENCE_THRESHOLD,
        help=f"MegaDetector confidence threshold (default: {config.DETECTION_CONFIDENCE_THRESHOLD})",
    )

    parser.add_argument(
        "--classifier-crop-padding-ratio",
        type=float,
        default=config.CROP_PADDING,
        help=f"Padding ratio used only for classifier crops (default: {config.CROP_PADDING})",
    )

    parser.add_argument(
        "--visualization-padding-ratio",
        type=float,
        default=0.0,
        help="Padding ratio used only for drawn boxes",
    )

    parser.add_argument(
        "--min-crop-width",
        type=int,
        default=config.MIN_CROP_WIDTH,
        help=f"Minimum classifier crop width in pixels (default: {config.MIN_CROP_WIDTH})",
    )

    parser.add_argument(
        "--min-crop-height",
        type=int,
        default=config.MIN_CROP_HEIGHT,
        help=f"Minimum classifier crop height in pixels (default: {config.MIN_CROP_HEIGHT})",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the final annotated image (single-file mode only)",
    )

    parser.add_argument(
        "--save-vis",
        type=str,
        default=None,
        help="Directory where annotated images will be saved",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file where structured results will be saved",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print anything to console",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed detection/classification output",
    )

    args = parser.parse_args()
    validate_args(args, parser)

    detector_model_path = Path(args.detector_model)
    classifier_model_path = Path(args.classifier_model)
    classes_path = Path(args.classes)
    save_vis_dir = Path(args.save_vis) if args.save_vis else None
    output_path = Path(args.output) if args.output else None
    classifier_img_size = (args.img_size[0], args.img_size[1])

    if not detector_model_path.is_file():
        raise FileNotFoundError(f"Detector model file not found: {detector_model_path}")

    if not classifier_model_path.is_file():
        raise FileNotFoundError(f"Classifier model file not found: {classifier_model_path}")

    classifier_model = load_model(classifier_model_path)
    class_names = get_class_names_from_dir(classes_path)

    num_model_outputs = classifier_model.output_shape[-1]
    if len(class_names) != num_model_outputs:
        raise ValueError(
            "Mismatch between class names and model outputs: "
            f"{len(class_names)} class names found, but model predicts {num_model_outputs} classes."
        )

    if args.file:
        image_path = Path(args.file)
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")

        result = detect_and_classify_single_image(
            image_path=image_path,
            detector_model_path=detector_model_path,
            classifier_model=classifier_model,
            class_names=class_names,
            detection_confidence_threshold=args.detection_threshold,
            classifier_img_size=classifier_img_size,
            classifier_crop_padding_ratio=args.classifier_crop_padding_ratio,
            visualization_padding_ratio=args.visualization_padding_ratio,
            min_crop_width=args.min_crop_width,
            min_crop_height=args.min_crop_height,
        )

        if not args.quiet:
            print_single_result(result, verbose=args.verbose)

        if args.show or save_vis_dir is not None:
            draw_detections_on_image(
                image_path=image_path,
                result=result,
                show=args.show,
                save_vis_dir=save_vis_dir,
            )

        if output_path is not None:
            save_results_to_json([result], output_path)

    else:
        dir_path = Path(args.dir)
        results = detect_and_classify_directory(
            dir_path=dir_path,
            detector_model_path=detector_model_path,
            classifier_model=classifier_model,
            class_names=class_names,
            detection_confidence_threshold=args.detection_threshold,
            classifier_img_size=classifier_img_size,
            classifier_crop_padding_ratio=args.classifier_crop_padding_ratio,
            visualization_padding_ratio=args.visualization_padding_ratio,
            min_crop_width=args.min_crop_width,
            min_crop_height=args.min_crop_height,
        )

        if save_vis_dir is not None:
            image_paths = find_image_files(dir_path)
            for image_path, result in zip(image_paths, results):
                draw_detections_on_image(
                    image_path=image_path,
                    result=result,
                    show=False,
                    save_vis_dir=save_vis_dir,
                )

        if not args.quiet:
            if args.verbose:
                for result in results:
                    print_single_result(result, verbose=True)
            else:
                print_directory_summary(results)

        if output_path is not None:
            save_results_to_json(results, output_path)


if __name__ == "__main__":
    main()
