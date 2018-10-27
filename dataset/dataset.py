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


def gen_parts_of_image(img: np.array, x_size: int, y_size: int, x_step: int, y_step: int) -> [np.array]:
    max_y, max_x = img.shape[:2]
    if max_y % y_size != 0:
        border_y = (y_size - (max_y % y_size) + 1) // 2
        img = cv2.copyMakeBorder(img, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        max_y = img.shape[0]
    if max_x % x_size != 0:
        border_x = (x_size - (max_x % x_size) + 1) // 2
        img = cv2.copyMakeBorder(img, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        max_x = img.shape[1]

    parts = []
    curr_y = 0
    while (curr_y + y_size) <= max_y:
        curr_x = 0
        while (curr_x + x_size) <= max_x:
            parts.append(img[curr_y:curr_y + y_size, curr_x:curr_x + x_size])
            curr_x += x_step
        curr_y += y_step
    return parts


def save_parts_of_image(file: str, x_size: int, y_size: int, x_step: int, y_step: int):
    dirname = file[:len(file) - 7] + '_parts'
    os.mkdir(dirname)
    in_parts = gen_parts_of_image(cv2.cvtColor(cv2.imread(file.replace('gt', 'in')), cv2.COLOR_BGR2GRAY),
                                  x_size, y_size, x_step, y_step)
    gt_parts = gen_parts_of_image(cv2.imread(file), x_size, y_size, x_step, y_step)
    for i in range(len(in_parts)):
        cv2.imwrite(dirname + '/' + str(i) + '_in.png', in_parts[i])
        cv2.imwrite(dirname + '/' + str(i) + '_gt.png', gt_parts[i])


descr_str = r"""
This script is designed to automate generating document image datasets.

It requires data folder with pairs of original and ground-truth images with following names format:
\d+_(gt|out).png (1_gt.png, 2_in.png, 33_gt.png, etc)."""


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(prog='dataset',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=descr_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-d', '--data', type=str, default=r'./data/',
                        help=r'path to data (default: ./data/)')
    parser.add_argument('--xsize', type=int, default=128,
                        help=r'x size of image part (default: %(default)s)')
    parser.add_argument('--ysize', type=int, default=128,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('--xstep', type=int, default=128,
                        help=r'x step (default: %(default)s)')
    parser.add_argument('--ystep', type=int, default=128,
                        help=r'y size of image part (default: %(default)s)')
    parser.add_argument('-g', '--gentrain', action='store_true',
                        help=r'generate training data')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(),
                        help=r'number of processes (default: %(default)s)')
    args = parser.parse_args()

    files = []
    for file in glob.iglob(args.data + '**/*_gt.png', recursive=True):
        files.append(file)
    Pool(args.processes).map(partial(save_parts_of_image, x_size=args.xsize, y_size=args.ysize,
                                     x_step=args.xstep, y_step=args.ystep), files)
    if args.gentrain:
        i = 0
        os.mkdir(args.data + 'dataset')
        for file in glob.iglob(args.data + '**/*_parts/*_gt.png', recursive=True):
            copy2(file.replace('gt', 'in'), args.data + 'dataset/' + str(i) + '_in.png')
            copy2(file, args.data + 'dataset/' + str(i) + '_gt.png')
            i += 1
        for file in glob.iglob(args.data + '**/*_parts', recursive=True):
            rmtree(file)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
