#!/usr/bin/python3

import argparse
import glob
import os
import time
from functools import partial
from multiprocessing import (Pool, cpu_count)
from shutil import (copy2, rmtree)

import cv2
import numpy as np


def split_img_overlay(img: np.array, size_x: int = 128, size_y: int = 128, step_x: int = 128, step_y: int = 128) -> [
    np.array]:
    """Split image to parts (little images) with possible overlay.

    Walk through the whole image by the window of size size_x * size_y with step step_x, step_y and
    save all parts in list. If the image sizes are not multiples of the window sizes,
    the image will be complemented by a frame of suitable size. If step_x, step_y are not equal to
    size_x, size_y, parts overlay each other, or have spaces between each other.

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
            curr_x += step_x
        curr_y += step_y
    return parts, border_y, border_x


def save_imgs(imgs_in: [np.array], imgs_gt: [np.array], fname_in: str):
    """Save image parts to one folder.

    Save all image parts to folder with name '(original image name) + _parts'.

    """
    dname = os.path.join(fname_in[:fname_in.rfind('_in')] + '_parts')
    mkdir_s(dname)
    for i, img in enumerate(imgs_in):
        cv2.imwrite(os.path.join(dname, str(i) + '_in.png'), img)
    for i, img in enumerate(imgs_gt):
        cv2.imwrite(os.path.join(dname, str(i) + '_gt.png'), img)


def process_img(fname_in, size_x: int = 128, size_y: int = 128, step_x: int = 128, step_y: int = 128):
    """Read train and groun_truth images, split them and save."""
    img_in = cv2.cvtColor(cv2.imread(fname_in), cv2.COLOR_BGR2GRAY)
    parts_in, _, _ = split_img_overlay(img_in, size_x, size_y, step_x, step_y)
    img_gt = cv2.cvtColor(cv2.imread(fname_in.replace('_in', '_gt')), cv2.COLOR_BGR2GRAY)
    parts_gt, _, _ = split_img_overlay(img_gt, size_x, size_y, step_x, step_y)
    save_imgs(parts_in, parts_gt, fname_in)


def shuffle_imgs(dname: str):
    """Shuffle input and groun-truth images (actual, if You are using different datasets as one)."""
    dir_in = os.path.join(dname, 'in')
    dir_gt = os.path.join(dname, 'gt')

    n = len(os.listdir(dir_in))
    np.random.seed()
    for i in range(n):
        j = i
        while j == i:
            j = np.random.randint(0, n)
        os.rename(os.path.join(dir_in, str(i) + '_in.png'), os.path.join(dir_in, 'tmp_in.png'))
        os.rename(os.path.join(dir_in, str(j) + '_in.png'), os.path.join(dir_in, str(i) + '_in.png'))
        os.rename(os.path.join(dir_in, 'tmp_in.png'), os.path.join(dir_in, str(j) + '_in.png'))
        os.rename(os.path.join(dir_gt, str(i) + '_gt.png'), os.path.join(dir_gt, 'tmp_gt.png'))
        os.rename(os.path.join(dir_gt, str(j) + '_gt.png'), os.path.join(dir_gt, str(i) + '_gt.png'))
        os.rename(os.path.join(dir_gt, 'tmp_gt.png'), os.path.join(dir_gt, str(j) + '_gt.png'))


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Create train and ground-truth images suitable for U-net from your dataset.

All train image names should end with "_in" like "1_in.png".
All ground-truth image should end with "_gt" like "1_gt.png".
If for some image there is only train or ground-truth version, script fails.
After script finishes, in the output directory there will be two subdirectories:
"in" with train images and "gt" with ground-truth images.

"""


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='dataset',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help=r'directory with input train and ground-truth images (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help=r'directory for output train and ground-truth images suitable for U-net (default: "%(default)s")')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help=r'shuffle images (actual, if You are using different datasets as one)')
    parser.add_argument('--size_x', type=int, default=128,
                        help=r'x size of image part (default: %(default)s)')
    parser.add_argument('--size_y', type=int, default=128,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('--step_x', type=int, default=128,
                        help=r'x step (default: %(default)s)')
    parser.add_argument('--step_y', type=int, default=128,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(),
                        help=r'number of processes (default: %(default)s)')
    return parser.parse_args()


def main():
    start_time = time.time()

    args = parse_args()

    fnames_in = list(glob.iglob(os.path.join(args.input, '**', '*_in.*'), recursive=True))
    f = partial(process_img, size_x=args.size_x, size_y=args.size_y, step_x=args.step_y, step_y=args.step_y)
    Pool(args.processes).map(f, fnames_in)
    mkdir_s(os.path.join(args.output))
    mkdir_s(os.path.join(args.output, 'in'))
    mkdir_s(os.path.join(args.output, 'gt'))
    for i, fname in enumerate(glob.iglob(os.path.join(args.input, '**', '*_parts', '*_in.*'), recursive=True)):
        copy2(os.path.join(fname), os.path.join(args.output, 'in', str(i) + '_in.png'))
        copy2(os.path.join(fname.replace('_in', '_gt')), os.path.join(args.output, 'gt', str(i) + '_gt.png'))
    for dname in glob.iglob(os.path.join(args.input, '**', '*_parts'), recursive=True):
        rmtree(dname)
    if args.shuffle:
        shuffle_imgs(args.output)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
