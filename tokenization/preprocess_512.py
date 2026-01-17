from PIL import Image

from data.transforms import LetterboxConfig, letterbox_512


def preprocess_image(img: Image.Image) -> Image.Image:
    return letterbox_512(img, LetterboxConfig())
