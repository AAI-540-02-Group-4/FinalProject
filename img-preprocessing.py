import cv2
import numpy as np
from config import IMG_SIZE


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


def read_and_preprocess_img(img_path) -> np.ndarray:

    # read img as grayscale
    img = cv2.imread(img_path, 0)

    # make sure img is in uin8
    norm_img = normalize_to_uint8(img)

    # apply clahe to enhance details
    enhanced_image = apply_clahe(norm_img)

    # resize img
    resized_img = resize_img(enhanced_image)

    return resized_img