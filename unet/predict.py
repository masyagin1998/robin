import argparse

from data import *
from model.unet import unet

descr_str = r"""
This script is designed to train U-net.

It requires data folder with pairs of original and ground-truth images with following names format:
\d+_(gt|out).png (1_gt.png, 2_in.png, 33_gt.png, etc)."""


def main():
    parser = argparse.ArgumentParser(prog='data.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=descr_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-d', '--data', type=str, default=r'./data/',
                        help=r'path to data (default: ./data/)')
    parser.add_argument('-w', '--weights', type=str, default='./data/train/in/',
                        help=r'x size of image part (default: %(default)s)')
    args = parser.parse_args()

    model = unet()
    model.load_weights(args.weights)
    testGene = testGenerator(args.data)
    results = model.predict_generator(testGene, 10, verbose=1)
    saveResult(args.data, results)


if __name__ == "__main__":
    main()
