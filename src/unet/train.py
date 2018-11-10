#!/usr/bin/python3

import argparse
import os
import shutil
import time

import cv2
import numpy as np
from alt_model_checkpoint import AltModelCheckpoint
from keras import backend as K
from keras.callbacks import (EarlyStopping, TensorBoard)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from model.unet import unet

GAUSSIAN_NOISE_MODE = 0


def gaussian_noise(img: np.array, mean: int, sigma: int) -> np.array:
    """Apply additive white gaussian noise to the image."""
    img = img.astype(np.int16)
    tmp = np.zeros(img.shape, np.int8)
    img = img + cv2.randn(tmp, mean, sigma)
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


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
        img[curr_y, curr_x] = 255
    for i in range(n // 2):
        # Pepper.
        curr_y = int(np.random.randint(0, h))
        curr_x = int(np.random.randint(0, w))
        img[curr_y, curr_x] = 0
    return img


CHANGE_BRIGHTNESS_MODE = 0


def change_brightness(img: np.array, diff: int) -> np.array:
    """Change brightness of image. If diff > 0 - brightness will be increased, else - decreased."""
    return img


CHANGE_CONTRAST_MODE = 1


def change_contrast(img: np.array) -> np.array:
    return img


def random_effect_img(img: np.array):
    """Add one of possible effects to image.

    Probability of noise effects:
    Gaussian noise    - 12,5%;
    Salt-pepper noise - 12,5%;
    No effects        - 75%;

    Probability of brightness/contrast effects:


    """
    i = np.random.randint(0, 8)
    if i == GAUSSIAN_NOISE_MODE:
        img = gaussian_noise(img, 0, 5)
    elif i == SALT_PEPPER_NOISE_MODE:
        img = salt_pepper_noise(img, 1)

    return img


def normalize_imgs(img_in: np.array, img_gt: np.array) -> (np.array, np.array):
    """Normalize image brightness to range [0.0..1.0]"""
    img_in = img_in / 255
    img_gt = img_gt / 255
    img_gt[img_gt > 0.5] = 1
    img_gt[img_gt <= 0.5] = 0
    return img_in, img_gt


def gen_data(dname: str, dataset_dname: str, start: int, stop: int, batch_size: int, augmentate: bool):
    """Generate images for training/validation/testing."""
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

    dir_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    ) if augmentate else ImageDataGenerator()

    dir_in_generator = dir_datagen.flow_from_directory(
        dname,
        classes=['in'],
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=1
    )

    dir_gt_generator = dir_datagen.flow_from_directory(
        dname,
        classes=['gt'],
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=1
    )
    dir_generator = zip(dir_in_generator, dir_gt_generator)
    for (img_in, img_gt) in dir_generator:
        img_in, img_gt = normalize_imgs(img_in, img_gt)
        if augmentate:
            img_in = random_effect_img(img_in)

        yield (img_in, img_gt)


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Train U-net with pairs of train and ground-truth images.

All train images should be in "in" directory.
All ground-truth images should be in "gt" directory.

"""


def parse_args():
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
    parser.add_argument('--train', type=int, default=80,
                        help=r'%% of train images (default: %(default)s%%)')
    parser.add_argument('--val', type=int, default=10,
                        help=r'%% of validation images (default: %(default)s%%)')
    parser.add_argument('--test', type=int, default=10,
                        help=r'%% of test images (default: %(default)s%%)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help=r'number of training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batchsize', type=int, default=20,
                        help=r'number of images, simultaneously sent to the GPU (default: %(default)s)')
    parser.add_argument('-g', '--gpus', type=int, default=1,
                        help=r'number of GPUs for training (default: %(default)s)')
    parser.add_argument('-a', '--augmentate', action='store_true',
                        help=r'use Keras data augmentation')
    parser.add_argument('-d', '--debug', type=str, default='',
                        help=r'directory to save tensorboard logs and weights history')
    return parser.parse_args()


smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def main():
    start_time = time.time()

    args = parse_args()

    input = args.input
    input_size = len(os.listdir(os.path.join(input, 'in')))

    tmp = args.tmp
    mkdir_s(tmp)

    if args.augmentate:
        np.random.seed()

    train_dir = os.path.join(tmp, 'train')
    train_start = 0
    train_stop = int(input_size * (args.train / 100))
    train_generator = gen_data(train_dir, input, train_start, train_stop,
                               args.batchsize * args.gpus, args.augmentate)

    validation_dir = os.path.join(tmp, 'validation')
    validation_start = train_stop
    validation_stop = validation_start + int(input_size * (args.val / 100))
    validation_generator = gen_data(validation_dir, input, validation_start, validation_stop,
                                    args.batchsize * args.gpus, args.augmentate)

    test_dir = os.path.join(tmp, 'test')
    test_start = validation_stop
    test_stop = input_size
    test_generator = gen_data(test_dir, input, test_start, test_stop,
                              args.batchsize * args.gpus, args.augmentate)

    original_model = unet()
    if args.gpus == 1:
        model = original_model
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=[jaccard_coef, 'accuracy'])
    else:
        model = multi_gpu_model(original_model, gpus=args.gpus)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=[jaccard_coef, 'accuracy'])

    callbacks = []
    if args.gpus == 1:
        model_checkpoint = AltModelCheckpoint(args.weights if args.debug == ''
                                              else os.path.join(args.debug, 'weights',
                                                                'weights-improvement-{epoch:02d}.hdf5'),
                                              model, monitor='val_jaccard_coef', mode='max', verbose=1,
                                              save_best_only=True, save_weights_only=True)
    else:
        model_checkpoint = AltModelCheckpoint(args.weights if args.debug == ''
                                              else os.path.join(args.debug, 'weights',
                                                                'weights-improvement-{epoch:02d}.hdf5'),
                                              original_model, monitor='val_jaccard_coef', mode='max', verbose=1,
                                              save_best_only=True, save_weights_only=True)
    callbacks.append(model_checkpoint)
    model_early_stopping = EarlyStopping(monitor='val_jaccard_coef', min_delta=0.001, patience=2, verbose=1, mode='max')
    callbacks.append(model_early_stopping)
    if args.debug != '':
        mkdir_s(args.debug)
        mkdir_s(os.path.join(args.debug, 'weights'))
        mkdir_s(os.path.join(args.debug, 'logs'))
        model_tensorboard = TensorBoard(log_dir=os.path.join(args.debug, 'logs'),
                                        histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(model_tensorboard)

    model.fit_generator(
        train_generator,
        steps_per_epoch=(train_stop - train_start + 1) / args.batchsize,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=(validation_stop - validation_start + 1) / args.batchsize,
        callbacks=callbacks
    )

    if args.debug != '':
        model.save_weights(args.weights)

    metrics = model.evaluate_generator(
        test_generator,
        steps=(test_stop - test_start + 1) / args.batchsize,
        verbose=1
    )
    print(metrics)

    shutil.rmtree(tmp)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
