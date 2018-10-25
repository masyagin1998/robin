#!/usr/bin/python3

import argparse
import os
import sys
import time
from functools import partial
from functools import reduce
from multiprocessing import Pool
from multiprocessing import cpu_count
from platform import system
from re import search


class Metrics:
    """
    Metrics contains basic DIBCO metrics for binarized and ground-truth image:
    F-Measure, pseudo F-Measure, PSNR, DRD.
    By default value of every measure is zero.
    """

    def __init__(self):
        self.fm = 0.0
        self.pfm = 0.0
        self.psnr = 0.0
        self.drd = 0.0

    def __add__(self, other):
        metrics = Metrics()
        metrics.fm = self.fm + other.fm
        metrics.pfm = self.pfm + other.pfm
        metrics.psnr = self.psnr + other.psnr
        metrics.drd = self.drd + other.drd
        return metrics

    def __str__(self):
        return 'FM:   {0:.2f};\nPFM:  {1:.2f};\nPSNR: {2:.2f};\nDRD:  {3:.2f};\n'.format(
            self.fm, self.pfm, self.psnr, self.drd)


def meter(filename: str, weights: str, metrics: str, data: str) -> Metrics:
    os.system(weights + ' ' + data + filename + " > NUL")
    os.system(metrics + ' ' +
              data + filename + ' ' +
              data + filename.replace('gt', 'out') + ' ' +
              data + filename.replace('.png', '_RWeights.dat') + ' ' +
              data + filename.replace('.png', '_PWeights.dat') + ' ' +
              '> ' + data + filename.replace('gt.png', 'res.txt'))
    os.remove(data + filename.replace('.png', '_RWeights.dat'))
    os.remove(data + filename.replace('.png', '_PWeights.dat'))
    m = Metrics()
    res_name = data + filename.replace("gt.png", "res.txt")
    with open(res_name, 'r') as res:
        dot_num_regexp = r'\d+\.\d+'
        for line in res:
            if 'pseudo F-Measure (Fps)' in line:
                m.pfm = float(search(dot_num_regexp, line).group(0))
            elif 'F-Measure' in line:
                m.fm = float(search(dot_num_regexp, line).group(0))
            elif 'PSNR' in line:
                m.psnr = float(search(dot_num_regexp, line).group(0))
            elif 'DRD' in line:
                m.drd = float(search(dot_num_regexp, line).group(0))
    with open(res_name, 'w') as res:
        res.write(str(m))
    return m


bad_os_str = r"""This script is platform-dependent.
It can be run only on Microsoft Windows."""

descr_str = r"""This script is designed to automate DIBCO measurement.

It requires DIBCO weights and metrics evaluation tools and data folder
with pairs of binarized and ground-truth images with following names format:
\d+_(gt|out).png (1_gt.png, 2_out.png, 33_gt.png, etc).

Output of script are text files in data folder with following names format:
\d+_res.png (1_res.txt, 33_res.txt) for every image and total_res.png for all folder.
Result files contain four measures: F-Measure, pseudo F-measure, PSNR and DRD."""


def main():
    if system() != 'Windows':
        print(bad_os_str)
        sys.exit(1)

    parser = argparse.ArgumentParser(prog='metrics.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=descr_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-w', '--weights', type=str, default=r'.\weights\weights.exe',
                        help=r'path to weights evaluation tool (default: %(default)s)')
    parser.add_argument('-m', '--metrics', type=str, default=r'.\metrics\metrics.exe',
                        help=r'path to metrics evaluation tool (default: %(default)s)')
    parser.add_argument('-d', '--data', type=str, default=r'.\data\\',
                        help=r'path to data (default: .\data\)')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(),
                        help=r'number of processes (default: %(default)s)')
    args = parser.parse_args()

    start_time = time.time()
    files = []
    for file in os.listdir(args.data):
        if file.endswith("_gt.png"):
            files.append(file)
    with open(args.data + 'total_res.txt', 'w') as res:
        total_res = reduce(lambda a, b: a + b, Pool(args.processes).map(
            partial(meter, weights=args.weights, metrics=args.metrics, data=args.data), files))
        n = len(files)
        total_res.fm /= n
        total_res.pfm /= n
        total_res.psnr /= n
        total_res.drd /= n
        res.write(str(total_res))
    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
