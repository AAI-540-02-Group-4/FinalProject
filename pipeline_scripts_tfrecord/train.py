"""Train the pneumonia CNN from pre-built TFRecord shards.

Channels mounted by SageMaker (local directories):
  /opt/ml/input/data/train/train.tfrecord
  /opt/ml/input/data/train/class_weights.json
  /opt/ml/input/data/validation/validation.tfrecord

Outputs:
  /opt/ml/model/pneumonia_cnn_model.keras
  /opt/ml/model/pneumonia_cnn_model.onnx
  /opt/ml/output/data/training_history.json
"""
import argparse
import json
import os
import pathlib

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

IMG_SIZE = 128

_feature_spec = {
    "image": tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE], tf.int64),
    "label": tf.io.FixedLenFeature([1], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_example(serialized):
    parsed = tf.io.parse_single_example(serialized, _feature_spec)
    img = tf.reshape(tf.cast(parsed["image"], tf.float32) / 255.0, (IMG_SIZE, IMG_SIZE, 1))
    label = tf.cast(parsed["label"][0], tf.float32)
    return img, label


def _build_dataset(tfrecord_path, batch_size, augment):
    ds = tf.data.TFRecordDataset(str(tfrecord_path), num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model():
    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return layers.MaxPooling2D()(x)

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = block(inp, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def main(args):
    train_tfrecord = pathlib.Path(args.train_dir) / "train.tfrecord"
    val_tfrecord = pathlib.Path(args.val_dir) / "validation.tfrecord"
    weights_path = pathlib.Path(args.train_dir) / "class_weights.json"

    with open(weights_path) as f:
        class_weight = {int(k): float(v) for k, v in json.load(f).items()}
    print(f"class_weight: {class_weight}")

    train_ds = _build_dataset(train_tfrecord, args.batch_size, augment=True)
    val_ds = _build_dataset(val_tfrecord, args.batch_size, augment=False)

    model = _build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, class_weight=class_weight,
        callbacks=cb,
    )

    out_model_dir = pathlib.Path(args.model_output_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_model_dir / "pneumonia_cnn_model.keras")

    # Additionally export in SavedModel format under a version-1 subdirectory.
    # The SageMaker TensorFlow Serving inference container requires this layout
    # (saved_model.pb + variables/ + assets/ at /opt/ml/model/<version>/).
    # Without it BatchTransform fails with "no SavedModel bundles found!".
    model.export(str(out_model_dir / "1"))

    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 1), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13,
            output_path=str(out_model_dir / "pneumonia_cnn_model.onnx"),
        )
        print("ONNX export OK")
    except Exception as e:
        print(f"ONNX export skipped: {type(e).__name__}: {e}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    out_data = pathlib.Path(args.output_data_dir)
    out_data.mkdir(parents=True, exist_ok=True)
    with open(out_data / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--train-dir", default=os.environ["SM_CHANNEL_TRAIN"])
    p.add_argument("--val-dir", default=os.environ["SM_CHANNEL_VALIDATION"])
    p.add_argument("--model-output-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    # SageMaker TF estimator always injects --model_dir on the command line; accept
    # and ignore it (we use --model-output-dir for our actual save location).
    p.add_argument("--model_dir", default=None)
    main(p.parse_args())
