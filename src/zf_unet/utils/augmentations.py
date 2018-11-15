import cv2
import numpy as np

GAUSSIAN_NOISE_MODE = 0


def gaussian_noise(img: np.array, mean: int, sigma: int) -> np.array:
    """Apply additive white gaussian noise to the image."""
    img = img.astype(np.int16)
    tmp = np.zeros(img.shape, np.int8)
    img = img + cv2.randn(tmp, mean, sigma)
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


SALT_PEPPER_NOISE_MODE = 1


def salt_pepper_noise(img: np.array, prop: int) -> np.array:
    """Apply "salt-and-pepper" noise to the image."""
    h = img.shape[0]
    w = img.shape[1]
    n = int(h * w * prop / 100)
    for i in range(n // 2):
        # Salt.
        curr_y = int(np.random.randint(0, h))
        curr_x = int(np.random.randint(0, w))
        img[curr_y, curr_x] = 255
    for i in range(n // 2):
        # Pepper.
        curr_y = int(np.random.randint(0, h))
        curr_x = int(np.random.randint(0, w))
        img[curr_y, curr_x] = 0
    return img


CHANGE_BRIGHTNESS_MODE = 0


def change_brightness(img: np.array, diff: int) -> np.array:
    """Change brightness of image. If diff > 0 - brightness will be increased, else - decreased."""
    return img


CHANGE_CONTRAST_MODE = 1


def change_contrast(img: np.array) -> np.array:
    return img


ELASTIC_TRANSFORM_MODE = 0


def elastic_transform(img: np.array) -> np.array:
    """Change image using elastic transform."""
    return img


def random_effect_img(img: np.array):
    """Add one of possible effects to image.

    Probability of noise effects:
    Gaussian noise    - 12,5%;
    Salt-pepper noise - 12,5%;
    No effects        - 75%;

    Probability of brightness/contrast effects:


    """
    i = np.random.randint(0, 8)
    if i == GAUSSIAN_NOISE_MODE:
        img = gaussian_noise(img, 0, 5)
    elif i == SALT_PEPPER_NOISE_MODE:
        img = salt_pepper_noise(img, 1)

    return img
