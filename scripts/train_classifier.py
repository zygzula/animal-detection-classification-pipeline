import argparse
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import src.config as config
from src.classification.model import build_model, compile_model
from src.utils.reproducibility_utils import set_seeds


def load_and_preprocess_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        pad_to_aspect_ratio=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        pad_to_aspect_ratio=True,
    )

    data_augmentation = build_data_augmentation()
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=autotune
    ).prefetch(buffer_size=autotune)

    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_data_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_callbacks() -> list[keras.callbacks.Callback]:
    config.CLASSIFIER_MODEL_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=config.CLASSIFIER_MODEL_BEST_PATH,
        monitor="val_loss",
        save_best_only=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1,
    )

    return [early_stopping, model_checkpoint, reduce_lr]


def fine_tune_model(
        model: keras.Model,
        base_model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        callbacks: list[keras.callbacks.Callback],
) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False

    # Keep BatchNorm frozen
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, learning_rate=config.FINE_TUNE_LEARNING_RATE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Animal classifier prediction")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output: lists of columns for the processed dataframes"
    )

    args = parser.parse_args()
    verbose = args.verbose

    set_seeds(config.SEED)

    train_ds, val_ds, class_names = load_and_preprocess_datasets()
    num_classes = len(class_names)

    print(f"Found {num_classes} classes.")
    if verbose:
        print(class_names)

    with open(config.CLASSIFIER_CLASS_MAPPING_PATH, "w") as f:
        json.dump(class_names, f)
    if verbose:
        print(f"Saved class mapping to {config.CLASSIFIER_CLASS_MAPPING_PATH}")

    # --- Model Building and Compilation ---
    model, base_model = build_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=config.DROPOUT_RATE,
    )
    compile_model(model, learning_rate=config.INITIAL_LEARNING_RATE)
    model.summary()

    print("\nTraining classifier head.")
    head_callbacks = build_callbacks()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_HEAD,
        callbacks=head_callbacks,
    )

    print("\nFine-tuning top layers.")
    fine_tune_callbacks = build_callbacks()
    fine_tune_model(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        callbacks=fine_tune_callbacks,
    )

    config.CLASSIFIER_MODEL_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.CLASSIFIER_MODEL_BEST_PATH)

    print(f"Best checkpoint saved to: {config.CLASSIFIER_MODEL_BEST_PATH}")


if __name__ == "__main__":
    main()
