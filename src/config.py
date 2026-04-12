"""
Centralized configuration for the animal classification project.
"""
from pathlib import Path

# =============================================================================
#  PATHS
# =============================================================================
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_BASE_PATH = PROJECT_ROOT / "logs"

# przeniesioone z prepare_training_data
TRAINING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "training_manifest.json"
CATEGORIES_PATH = PROJECT_ROOT / "models" / "categories.json"
MODEL_PATH = PROJECT_ROOT / "models" / "md_v5a.0.0.pt"

IMAGE_INPUT_DIR = PROJECT_ROOT / "data" / "interim"
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "train"

BATCH_SIZE = 16
DETECTION_CONFIDENCE_THRESHOLD = 0.2
LOGS_FILENAME = "download.log"
# koniec prepare_training_data

# this goes from download.py
BASE_IMAGE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped"

REQUEST_TIMEOUT = 10
VERBOSE = False
#end of download.py

# Path to the training dataset of cropped images
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"

# --- Model Paths ---
MODEL_DIR = PROJECT_ROOT / "models"

# MegaDetector model file
DETECTOR_MODEL_PATH = MODEL_DIR / "md_v5a.0.0.pt"

# Classifier model files
CLASSIFIER_MODEL_BEST_PATH = MODEL_DIR / "animal_classifier.keras"
CLASSIFIER_CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"



PREDICTION_OUTPUT_PATH = PROJECT_ROOT / "outputs"


# =============================================================================
#  REPRODUCIBILITY
# =============================================================================
# Seed for all random operations to ensure reproducibility
SEED = 124


# =============================================================================
#  DETECTION & CROPPING
# =============================================================================
# Confidence threshold for MegaDetector detections
DETECTOR_CONFIDENCE = 0.8

# Padding to add around the cropped bounding box (as a fraction of width/height)
# This helps provide context to the classifier.
CROP_PADDING = 0.08

# Categories to consider for cropping (from MegaDetector output)
# 1: animal, 2: person, 3: vehicle
DETECTOR_ANIMAL_CATEGORY = "1"


# =============================================================================
#  CLASSIFIER TRAINING
# =============================================================================
# --- Dataset Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # 20% of the data will be used for validation


# final detection script
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTOR_CATEGORY_MAP = {
    "1": "animal",
    "2": "person",
    "3": "vehicle",
}

MIN_CROP_WIDTH = 32
MIN_CROP_HEIGHT = 32
# end

# --- Model Architecture ---
# For MobileNetV2, the input shape must be 3-channel
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
DROPOUT_RATE = 0.2

# --- Transfer Learning & Fine-Tuning ---
# Number of epochs for training the new classifier head (backbone frozen)
EPOCHS_HEAD = 50
INITIAL_LEARNING_RATE = 1e-3

# Number of epochs for fine-tuning (backbone partially unfrozen)
EPOCHS_FINE_TUNE = 50
FINE_TUNE_LEARNING_RATE = 5e-6

# Layer index from which to start unfreezing the base model for fine-tuning
FINE_TUNE_AT = 50

# --- Callbacks ---
# Patience for EarlyStopping callback
EARLY_STOPPING_PATIENCE = 3
# Factor for ReduceLROnPlateau callback
REDUCE_LR_FACTOR = 0.3
# Patience for ReduceLROnPlateau callback
REDUCE_LR_PATIENCE = 2


# =============================================================================
#  INFERENCE
# =============================================================================
# Number of top predictions to show
TOP_K_PREDICTIONS = 3
