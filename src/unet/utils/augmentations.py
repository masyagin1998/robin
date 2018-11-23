import cv2
import numpy as np

GAUSSIAN_NOISE_MODE = 0


def gaussian_noise(img: np.array, mean: int, sigma: int) -> np.array:
    """Apply additive white gaussian noise to the image."""
    tmp = np.zeros(img.shape, np.float32)
    img = img + cv2.randn(tmp, mean, sigma)
    img[img < 0.0] = 0.0
    img[img > 255.0] = 255.0
    return img


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
        img[curr_y, curr_x] = 255.0
    for i in range(n // 2):
        # Pepper.
        curr_y = int(np.random.randint(0, h))
        curr_x = int(np.random.randint(0, w))
        img[curr_y, curr_x] = 0.0
    return img


def random_effect_img(img_in: np.array, img_gt: np.array) -> (np.array, np.array):
    """Add one of possible effects to image.

    Probability of noise effects:
    Gaussian noise    - 25%;
    Salt-pepper noise - 25%;
    No effects        - 50%;
    """

    # Noise.
    i = np.random.randint(0, 4)
    if i == GAUSSIAN_NOISE_MODE:
        img_in = gaussian_noise(img_in, 0.0, 10.0)
    elif i == SALT_PEPPER_NOISE_MODE:
        img_in = salt_pepper_noise(img_in, 1)

    return img_in, img_gt
