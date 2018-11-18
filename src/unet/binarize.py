#!/usr/bin/python3

import argparse
import glob
import time

from keras.optimizers import Adam

from model.unet import unet
from utils.img_processing import *

desc_str = r"""Binarize images from input directory and write them to output directory.

All input image names should end with "_in" like "1_in.png".
All output image names will end with "_out" like "1_out.png".

"""


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='binarize',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help=r'directory with input images (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help=r'directory for output images (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join('.', 'bin_weights.hdf5'),
                        help=r'path to U-net weights (default: "%(default)s")')
    parser.add_argument('-b', '--batchsize', type=int, default=20,
                        help=r'number of images, simultaneously sent to the GPU (default: %(default)s)')
    parser.add_argument('-g', '--gpus', type=int, default=1,
                        help=r'number of GPUs for binarization (default: %(default)s)')
    return parser.parse_args()


def main():
    start_time = time.time()

    args = parse_args()

    fnames_in = list(glob.iglob(os.path.join(args.input, '**', '*_in.*'), recursive=True))
    model = None
    if len(fnames_in) != 0:
        mkdir_s(args.output)
        model = unet()
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(args.weights)
    for fname in fnames_in:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = binarize_img(img, model, args.batchsize)
        cv2.imwrite(os.path.join(args.output, os.path.split(fname)[-1].replace('_in', '_out')), img)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
