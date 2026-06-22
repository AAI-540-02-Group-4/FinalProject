"""Evaluate the trained CNN against the test TFRecord shard.

Inputs:  /opt/ml/processing/model/model.tar.gz
         /opt/ml/processing/test/test.tfrecord
Output:  /opt/ml/processing/evaluation/evaluation.json
"""
import json
import os
import pathlib
import tarfile

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

IMG_SIZE = 128
THRESHOLD = 0.4

_feature_spec = {
    "image": tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE], tf.int64),
    "label": tf.io.FixedLenFeature([1], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_example(serialized):
    parsed = tf.io.parse_single_example(serialized, _feature_spec)
    img = tf.reshape(tf.cast(parsed["image"], tf.float32) / 255.0, (IMG_SIZE, IMG_SIZE, 1))
    label = tf.cast(parsed["label"][0], tf.int32)
    return img, label


def main():
    model_tar = pathlib.Path("/opt/ml/processing/model/model.tar.gz")
    extract_to = pathlib.Path("/tmp/model")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar) as t:
        t.extractall(extract_to)

    keras_path = next(extract_to.rglob("pneumonia_cnn_model.keras"), None)
    if keras_path is None:
        sm_dir = next(p.parent for p in extract_to.rglob("saved_model.pb"))
        model = tf.keras.models.load_model(sm_dir)
    else:
        model = tf.keras.models.load_model(keras_path)

    test_tfrecord = "/opt/ml/processing/test/test.tfrecord"
    ds = tf.data.TFRecordDataset(test_tfrecord).map(_parse_example).batch(64)
    print(f"Evaluating from {test_tfrecord} at threshold={THRESHOLD}")

    y_prob_chunks, y_true_chunks = [], []
    for imgs, labels in ds:
        y_prob_chunks.append(model.predict(imgs, verbose=0).reshape(-1))
        y_true_chunks.append(labels.numpy())
    y_prob = np.concatenate(y_prob_chunks)
    y_true = np.concatenate(y_true_chunks).astype(int)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = {
        "binary_classification_metrics": {
            "accuracy":  {"value": float(accuracy_score(y_true, y_pred)),  "standard_deviation": "NaN"},
            "precision": {"value": float(precision_score(y_true, y_pred)), "standard_deviation": "NaN"},
            "recall":    {"value": float(recall_score(y_true, y_pred)),    "standard_deviation": "NaN"},
            "f1":        {"value": float(f1_score(y_true, y_pred)),        "standard_deviation": "NaN"},
            "auc":       {"value": float(roc_auc_score(y_true, y_prob)),   "standard_deviation": "NaN"},
            "confusion_matrix": cm,
            "threshold": THRESHOLD,
        }
    }
    print(json.dumps(report, indent=2))

    out = pathlib.Path("/opt/ml/processing/evaluation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation.json", "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
