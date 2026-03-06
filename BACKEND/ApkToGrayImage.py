import math
import numpy as np
from PIL import Image

def apk_bytes_to_gray_image(apk_bytes: bytes) -> Image.Image:
    """
    Thesis-style common approach:
    APK bytes -> uint8 -> pad -> reshape square -> grayscale image.
    """
    arr = np.frombuffer(apk_bytes, dtype=np.uint8)
    side = int(math.ceil(math.sqrt(arr.size)))
    padded = np.pad(arr, (0, side * side - arr.size), mode="constant", constant_values=0)
    img2d = padded.reshape((side, side))
    return Image.fromarray(img2d, mode="L")
