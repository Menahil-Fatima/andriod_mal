from PIL import Image

def resize_gray(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)
