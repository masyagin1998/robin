#!/usr/bin/python3

import argparse
import glob
import os
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count

import cv2
import numpy as np


def gen_parts_of_image(img: np.array, x_size: int, y_size: int, x_step: int, y_step: int) -> [np.array]:
    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    while (curr_y + y_size) < max_y:
        curr_x = 0
        while (curr_x + x_size) < max_x:
            parts.append(img[curr_y:curr_y + y_size, curr_x:curr_x + x_size])
            curr_x += x_step
        curr_y += y_step
    return parts


def save_parts_of_image(file: str, x_size: int, y_size: int, x_step: int, y_step: int):
    os.mkdir(file[:len(file) - 7])
    print(file)
    in_parts = gen_parts_of_image(cv2.imread(file.replace('gt', 'in'), cv2.COLOR_RGB2GRAY),
                                  x_size, y_size, x_step, y_step)
    gt_parts = gen_parts_of_image(cv2.imread(file), x_size, y_size, x_step, y_step)
    for i in range(len(in_parts)):
        cv2.imwrite(file[:len(file) - 7] + '/' + str(i) + '_in.png', in_parts[i])
        cv2.imwrite(file[:len(file) - 7] + '/' + str(i) + '_gt.png', gt_parts[i])


descr_str = r"""
This script is designed to automate generating document image datasets.

It requires data folder with pairs of original and ground-truth images with following names format:
\d+_(gt|out).png (1_gt.png, 2_in.png, 33_gt.png, etc)."""


def main():
    parser = argparse.ArgumentParser(prog='data.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=descr_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-d', '--data', type=str, default=r'./data/',
                        help=r'path to data (default: ./data/)')
    parser.add_argument('--xsize', type=int, default=256,
                        help=r'x size of image part (default: %(default)s)')
    parser.add_argument('--ysize', type=int, default=256,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('--xstep', type=int, default=256,
                        help=r'x step (default: %(default)s)')
    parser.add_argument('--ystep', type=int, default=256,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(),
                        help=r'number of processes (default: %(default)s)')
    args = parser.parse_args()

    start_time = time.time()
    files = []
    for file in glob.iglob(args.data + '**/*_gt.png', recursive=True):
        files.append(file)
    Pool(args.processes).map(partial(save_parts_of_image, x_size=args.xsize, y_size=args.ysize,
                                     x_step=args.xstep, y_step=args.ystep), files)
    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
