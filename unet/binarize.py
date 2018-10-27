#!/usr/bin/python3

import argparse
import os
import time

import cv2
import numpy as np


def split_img(img: np.array, size_x: int = 128, size_y: int = 128) -> ([np.array], int, int):
    """Split image to parts (little images).

    Walk through the whole image by the window of size size_x * size_y without overlays and
    save all parts in list. If the image sizes are not multiples of the window sizes,
    the image will be complemented by a frame of suitable size.

    """
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = (size_y - (max_y % size_y) + 1) // 2
        img = cv2.copyMakeBorder(img, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        max_y = img.shape[0]
    border_x = 0
    if max_x % size_x != 0:
        border_x = (size_x - (max_x % size_x) + 1) // 2
        img = cv2.copyMakeBorder(img, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        max_x = img.shape[1]
    parts = []
    curr_y = 0
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += size_x
        curr_y += size_y
    return parts, border_y, border_x


def combine_imgs(imgs: [np.array], border_y: int, border_x: int, max_y: int, max_x: int) -> np.array:
    """Combine little images to one big image.

    Walk through list of images and create from them one big image with sizes max_x * max_y.
    If border_x and border_y are non-zero, they will be removed from created image.
    The list of images should contain data in the following order:
    from left to right, from top to bottom.

    """
    max_y += (border_y * 2)
    max_x += (border_x * 2)
    img = np.zeros((max_y, max_x), np.uint8)
    size_y, size_x = imgs[0].shape
    curr_y = 0
    i = 0
    while (curr_y + size_y) <= max_y + border_y * 2:
        curr_x = 0
        while (curr_x + size_x) <= max_x + border_x * 2:
            img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imgs[i]
            i += 1
            curr_x += size_x
        curr_y += size_y
    img = img[border_y:img.shape[0] - border_y, border_x:img.shape[1] - border_x]
    return img


def normalize_img(img: np.array) -> np.array:
    """Normalize image channels from uint[0..255] to float[0.0..1.0]."""
    return img.astype(float) / 255


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Binarize images from input directory and write them to output directory.

All input image names should end with "_in" like "1_in.png".
All output image names will end with "_out" like "1_out.png".

"""


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='binarize',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-i', '--input', type=str, default=r'./input/',
                        help=r'directory with input images (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=str, default=r'./output/',
                        help=r'directory for output images (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default=r'./bin_weights.hdf5',
                        help=r'path to U-net weights (default: "%(default)s")')
    args = parser.parse_args()

    fnames = os.listdir(args.input)
    if len(fnames) != 0:
        mkdir_s(args.output)
    for fname in fnames:
        img = cv2.cvtColor(cv2.imread(os.path.join(args.input, fname)), cv2.COLOR_BGR2GRAY)
        parts, border_y, border_x = split_img(img)
        parts = np.array(parts)
        parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2], 1)
        parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2])
        img = combine_imgs(parts, border_y, border_x, img.shape[0], img.shape[1])
        cv2.imwrite(os.path.join(args.output, '{}_out.png'.format(fname[:fname.rfind('_in.')])), img)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
