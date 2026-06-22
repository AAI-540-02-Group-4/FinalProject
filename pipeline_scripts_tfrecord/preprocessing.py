"""Split the project manifest 40/10/10/40 and materialize TFRecord shards.

Reads:  /opt/ml/processing/input/image_metadata.csv
Outputs (per split, where <split> in {train, validation, test, production}):
    /opt/ml/processing/<split>/<split>.tfrecord
    /opt/ml/processing/train/class_weights.json

Each TFRecord example stores:
    image:    int64 [128*128] grayscale uint8 values (decoded + resized here)
    label:    int64 scalar (0=NORMAL, 1=PNEUMONIA)
    image_id: bytes (for traceability)

Pre-decoding once here saves the training step from doing 33K per-image
S3 reads + PNG decodes every epoch.
"""
import subprocess
import sys

# The TensorFlow ScriptProcessor image does NOT ship scikit-learn or Pillow.
# Install them at startup; we need sklearn for the stratified split and Pillow
# for PNG decode/resize before TFRecord write. ~5s of cold-start time.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "scikit-learn", "Pillow"])

import argparse
import io
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 128
_s3 = boto3.client("s3")


def _parse_s3(uri_or_key, default_bucket):
    """Accept either s3://bucket/key or a bare key.

    The canonical manifest now publishes full s3:// URIs (see
    data_preparations.ipynb §3). The bare-key branch is kept as a
    defensive fallback for legacy manifests."""
    if uri_or_key.startswith("s3://"):
        rest = uri_or_key[len("s3://"):]
        b, _, k = rest.partition("/")
        return b, k
    return default_bucket, uri_or_key.lstrip("/")


def _load_and_resize(s3_uri_or_key, default_bucket):
    bucket, key = _parse_s3(s3_uri_or_key, default_bucket)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    img = Image.open(io.BytesIO(obj["Body"].read())).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _make_example(arr_uint8, label_int, image_id):
    feature = {
        "image": tf.train.Feature(int64_list=tf.train.Int64List(value=arr_uint8.flatten().tolist())),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_int)])),
        "image_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_shard(frame, out_path, default_bucket, workers=16):
    """Write one TFRecord shard. Parallel S3 reads, serial writes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = frame.to_dict(orient="records")

    def _fetch(row):
        try:
            arr = _load_and_resize(row["preprocessed_s3_key"], default_bucket)
            return row["image_id"], int(row["label_int"]), arr
        except Exception as e:
            return row["image_id"], None, str(e)

    written = errors = 0
    with tf.io.TFRecordWriter(str(out_path)) as writer:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch, r) for r in rows]
            for i, fut in enumerate(as_completed(futures), start=1):
                image_id, label, payload = fut.result()
                if label is None:
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR fetching {image_id}: {payload}")
                    continue
                writer.write(_make_example(payload, label, image_id).SerializeToString())
                written += 1
                if i % 1000 == 0:
                    print(f"  {out_path.name}: {i}/{len(rows)}")
    print(f"  {out_path.name}: {written} examples ({errors} errors)")


def main(args):
    base = pathlib.Path("/opt/ml/processing")
    df = pd.read_csv(base / "input" / "image_metadata.csv")
    if "preprocessed_s3_key" not in df.columns:
        raise ValueError("Manifest missing 'preprocessed_s3_key' — run data_preparations.ipynb §5.4 first.")
    df = df.dropna(subset=["label", "label_int", "preprocessed_s3_key"]).reset_index(drop=True)
    print(f"Input rows: {len(df)}; class counts: {df['label'].value_counts().to_dict()}")

    df_model, df_prod = train_test_split(
        df, test_size=0.40, random_state=args.random_state, stratify=df["label"]
    )
    df_train, df_temp = train_test_split(
        df_model, test_size=0.333, random_state=args.random_state, stratify=df_model["label"]
    )
    df_test, df_val = train_test_split(
        df_temp, test_size=0.5, random_state=args.random_state, stratify=df_temp["label"]
    )

    splits = {"train": df_train, "validation": df_val, "test": df_test, "production": df_prod}
    for name, frame in splits.items():
        print(f"\n=== {name}: {len(frame)} rows ===")
        write_shard(
            frame,
            base / name / f"{name}.tfrecord",
            default_bucket=args.bucket,
            workers=args.workers,
        )

    # compute_class_weight requires a numpy.ndarray for `classes` in newer sklearn,
    # not a Python list — wrap with np.asarray.
    classes = np.asarray(sorted(df_train["label_int"].unique()), dtype=np.int64)
    weights = compute_class_weight("balanced", classes=classes, y=df_train["label_int"].to_numpy())
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    with open(base / "train" / "class_weights.json", "w") as f:
        json.dump(class_weight, f)
    print(f"\nclass_weight: {class_weight}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--bucket", type=str, required=True,
                   help="Fallback bucket for legacy bare-key manifests; canonical "
                        "manifests publish full s3:// URIs and this is unused.")
    p.add_argument("--workers", type=int, default=16, help="Parallel S3-fetch threads")
    main(p.parse_args())
