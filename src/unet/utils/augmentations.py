import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

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


def random_effect_img(img_in: np.array, img_gt: np.array) -> (np.array, np.array):
    """Add one of possible effects to image.

    Probability of noise effects:
    Gaussian noise    - 12,5%;
    Salt-pepper noise - 12,5%;
    No effects        - 75%;

    Probability of brightness/contrast effects:


    """

    # Noise.
    i = np.random.randint(0, 4)
    if i == GAUSSIAN_NOISE_MODE:
        img_in = gaussian_noise(img_in, 0.0, 10.0)
    elif i == SALT_PEPPER_NOISE_MODE:
        img_in = salt_pepper_noise(img_in, 1)

    # Elastic transformation.

    return img_in, img_gt


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
