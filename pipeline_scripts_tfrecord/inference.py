"""SageMaker inference handler for the pneumonia CNN.

The TF Serving container that BatchTransform / endpoints use can't natively
decode PNG bytes — it expects JSON-formatted tensors. This handler lets us
keep the input as `application/x-image` PNG bytes (which is what
preprocessed-images/ contains) by decoding to a float32 tensor on the way in.

Handler contract (SageMaker TF Serving):
    input_handler(data, context)  → request body sent to TF Serving's REST API
    output_handler(data, context) → response body returned to the caller
"""
import io
import json

import numpy as np
from PIL import Image

IMG_SIZE = 128


def _to_bytes(data):
    """SageMaker hands `data` to input_handler in different shapes depending on the
    SDK version and content type. Normalize it to raw bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        # If the wrapper pre-decoded the body to text, it used latin-1 so bytes
        # round-trip safely (every byte 0-255 maps to a codepoint).
        return data.encode("latin-1")
    # File-like object (older SDK shape).
    return data.read()


def input_handler(data, context):
    """Decode incoming bytes → JSON payload TF Serving expects.

    For BatchTransform with `content_type="application/x-image"`, `data` arrives
    as PNG bytes (sometimes wrapped, sometimes raw — see _to_bytes). We decode the
    PNG, resize to IMG_SIZE, normalize to [0, 1], and wrap as TF Serving's
    `instances` JSON.
    """
    ct = context.request_content_type

    if ct == "application/json":
        # Already JSON-formatted — pass through.
        raw = _to_bytes(data)
        return raw.decode("utf-8")

    if ct in ("application/x-image", "image/png"):
        img_bytes = _to_bytes(data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(IMG_SIZE, IMG_SIZE, 1)
        return json.dumps({"instances": [arr.tolist()]})

    raise ValueError(f"Unsupported content type: {ct}")


def output_handler(response, context):
    """Pass TF Serving's JSON response back to the caller unchanged."""
    if response.status_code != 200:
        raise ValueError(f"Inference error: {response.status_code} {response.content!r}")
    return response.content, context.accept_header
