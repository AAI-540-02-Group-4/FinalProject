import cv2
import numpy as np
from cloudpathlib import S3Path
import pydicom
from config import IMG_SIZE
import boto3

s3 = boto3.client("s3")


def read_s3_img(s3_img_path: S3Path) -> np.ndarray | None:
    """
    input: s3 object file path
    should handle reading in .jpeg or .dcm image files

    returns: image array
    """
    if type(s3_img_path) is not S3Path:
        raise ValueError(f'Invalid file type passed to function: {type(s3_img_path)}. Expected type = S3Path')

    if s3_img_path.suffix == '.dcm':
        with s3_img_path.open('rb') as f:
            dcm = pydicom.dcmread(f)
        img = dcm.pixel_array
        return img

    elif s3_img_path.suffix == '.jpeg':
        img = cv2.imread(s3_img_path, 0)
        return img


def load_image_from_s3(s3_key, bucket):
    """Load image from S3 - handles both JPEG and DICOM"""
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    img_bytes = response["Body"].read()

    if s3_key.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(img_bytes))
        img = ds.pixel_array
    else:
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    return img.astype(float32)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    this function takes and image and normalizes it to the min and max of 
    0 and 255. Then transoforms the type to 8 bit
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)



def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(img)
    return enhanced_image



def resize_img(img):
    return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)


def read_and_preprocess_img(s3_img_path) -> np.ndarray:

    # read img 
    img = read_s3_img(s3_img_path)

    if img is None:
        raise ValueError(f'Unable to open this image: {s3_img_path}')

    # convert to grayscale if there is more than two channels
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # make sure img is in uin8
    norm_img = normalize_to_uint8(img)

    # apply clahe to enhance details
    enhanced_image = apply_clahe(norm_img)

    # resize img
    resized_img = resize_img(enhanced_image)

    return resized_img