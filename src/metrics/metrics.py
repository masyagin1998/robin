#!/usr/bin/python3

import argparse
import glob
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
    """Metrics contains basic DIBCO metrics for binarized and ground-truth image:
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


def meter(fname_out: str, weights: str, metrics: str, data: str) -> Metrics:
    """Meter F-Measure, pseudo F-Measure, PSNR and DRD of your binarization."""
    fname_gt = fname_out.replace('_out', '_gt')
    fname_res = fname_gt[:fname_gt.rfind('_gt')] + '_res.txt'
    os.system(weights + ' ' + os.path.join(data, fname_gt) + " > NUL")
    os.system(metrics + ' ' +
              os.path.join(data, fname_gt) + ' ' +
              os.path.join(data, fname_out) + ' ' +
              os.path.join(data, fname_gt[:fname_gt.rfind('.')] + '_RWeights.dat') + ' ' +
              os.path.join(data, fname_gt[:fname_gt.rfind('.')] + '_PWeights.dat') + ' ' +
              '> ' + os.path.join(data, fname_res))
    os.remove(os.path.join(data, fname_gt[:fname_gt.rfind('.')] + '_RWeights.dat'))
    os.remove(os.path.join(data, fname_gt[:fname_gt.rfind('.')] + '_PWeights.dat'))
    m = Metrics()
    with open(os.path.join(data, fname_res), 'r') as res:
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
    with open(os.path.join(data, fname_res), 'w') as res:
        res.write(str(m))
    return m


bad_os_str = r"""This script is platform-dependent. It can be run only on Microsoft Windows."""

desc_str = r"""Meter quality of your binarization.

Only for Microsoft Windows.

Script requires DIBCO weights and metrics evaluation tools and
directory with input binarized and ground-truth images.
All binarized image names should end with "_out" like "1_out.png".
All ground-truth image should end with "_gt" like "1_gt.png".
After script finishes, in the output directory there will be metrics files
wtih four measures: F-Measure, pseudo F-Measure, PSNR and DRD.

"""


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='metrics',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help=r'directory with input binarized and ground-truth images (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help=r'directory with output metrics files (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join('weights', 'weights.exe'),
                        help=r'path to weights evaluation tool (default: %(default)s)')
    parser.add_argument('-m', '--metrics', type=str, default=os.path.join('metrics', 'metrics.exe'),
                        help=r'path to metrics evaluation tool (default: %(default)s)')
    parser.add_argument('-p', '--procs', type=int, default=cpu_count(),
                        help=r'number of processes (default: %(default)s)')
    return parser.parse_args()


def main():
    start_time = time.time()

    if system() != 'Windows':
        print(bad_os_str)
        sys.exit(1)

    args = parse_args()

    fnames_out = list(glob.iglob(os.path.join(args.input, '**', '*_out.*'), recursive=True))
    with open(os.path.join(args.output, 'total_res.txt'), 'w') as res:
        total_res = reduce(lambda a, b: a + b, Pool(args.processes).map(
            partial(meter, weights=args.weights, metrics=args.metrics, data=args.data), fnames_out))
        n = len(fnames_out)
        total_res.fm /= n
        total_res.pfm /= n
        total_res.psnr /= n
        total_res.drd /= n
        res.write(str(total_res))

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
