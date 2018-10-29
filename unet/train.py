#!/usr/bin/python3

import argparse
import os
import shutil
import time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from model.unet import unet


def adjustData(img, mask):
    if (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def gen_data(dname: str, dataset_dname: str, start: int, stop: int):
    os.mkdir(dname)
    dir_in = os.path.join(dname, 'in')
    os.mkdir(dir_in)
    dir_gt = os.path.join(dname, 'gt')
    os.mkdir(dir_gt)
    fnames = ['{}_in.png'.format(i) for i in range(start, stop)]
    for fname in fnames:
        src = os.path.join(dataset_dname, 'in', fname)
        dst = os.path.join(dir_in, fname)
        shutil.copy2(src, dst)
        src = os.path.join(dataset_dname, 'gt', fname.replace('in', 'gt'))
        dst = os.path.join(dir_gt, fname)
        shutil.copy2(src, dst)
    dir_datagen = ImageDataGenerator()
    dir_in_generator = dir_datagen.flow_from_directory(
        dname,
        classes=['in'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale',
        seed=1
    )

    dir_gt_generator = dir_datagen.flow_from_directory(
        dname,
        classes=['gt'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale',
        seed=1
    )
    dir_generator = zip(dir_in_generator, dir_gt_generator)
    for (img_in, img_gt) in dir_generator:
        img_in, img_gt = adjustData(img_in, img_gt)
        yield (img_in, img_gt)


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Train U-net with pairs of train and ground-truth images.

All train images should be in "in" directory.
All ground-truth images should be in "gt" directory.

"""


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='train',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help=r'directory with input train and ground-truth images (default: "%(default)s")')
    parser.add_argument('-t', '--tmp', type=str, default=os.path.join('.', 'tmp'),
                        help=r'directory for temporary training files. It will be deleted after script finishes (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join('.', 'bin_weights.hdf5'),
                        help=r'output U-net weights file (default: "%(default)s")')
    parser.add_argument('--train', type=int, default=80, help=r'% of train images (default: %(default)s%)')
    parser.add_argument('--val', type=int, default=20, help=r'% of validation images (default: %(default)s%)')
    args = parser.parse_args()

    input = args.input
    input_size = len(os.listdir(os.path.join(input, 'in')))

    tmp = args.tmp
    mkdir_s(tmp)

    train_dir = os.path.join(tmp, 'train')
    train_start = 0
    train_stop = int(input_size * (args.train / 100))
    train_generator = gen_data(train_dir, input, train_start, train_stop)

    validation_dir = os.path.join(tmp, 'validation')
    validation_start = train_stop
    validation_stop = input_size
    validation_generator = gen_data(validation_dir, input, validation_start, validation_stop)

    model = unet()
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=30
    )

    model.save_weights(args.weights)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
