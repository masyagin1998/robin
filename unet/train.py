#!/usr/bin/python3

import os
import shutil

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


def main():
    dataset_dir = './input/'
    dataset_size = len(os.listdir(dataset_dir + 'in'))
    print(dataset_size)

    base_dir = './data/'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    train_percents = 80
    train_start = 0
    train_stop = int(dataset_size * (train_percents / 100))
    train_generator = gen_data(train_dir, dataset_dir, train_start, train_stop)

    validation_dir = os.path.join(base_dir, 'validation')
    validation_percents = 20
    validation_start = train_stop
    validation_stop = validation_start + int(dataset_size * (validation_percents / 100))
    validation_generator = gen_data(validation_dir, dataset_dir, validation_start, validation_stop)

    model = unet()
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=30
    )

    model.save_weights('bin_weights.h5')


if __name__ == "__main__":
    main()
