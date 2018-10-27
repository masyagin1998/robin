#!/usr/bin/python3

import os
import shutil

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from model.unet import unet


def gen_data(dir_name: str, dataset_dir_name: str, start: int, stop: int):
    os.mkdir(dir_name)
    dir_in = os.path.join(dir_name, 'in')
    os.mkdir(dir_in)
    dir_gt = os.path.join(dir_name, 'gt')
    os.mkdir(dir_gt)
    fnames = ['{}_in.png'.format(i) for i in range(start, stop)]
    for fname in fnames:
        src = os.path.join(dataset_dir_name, fname)
        dst = os.path.join(dir_in, fname)
        shutil.copy2(src, dst)
        src = os.path.join(dataset_dir_name, fname.replace('in', 'gt'))
        dst = os.path.join(dir_gt, fname)
        shutil.copy2(src, dst)
    dir_datagen = ImageDataGenerator(rescale=1. / 255)
    dir_in_generator = dir_datagen.flow_from_directory(
        dir_name,
        classes=['in'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale'
    )

    dir_gt_generator = dir_datagen.flow_from_directory(
        dir_name,
        classes=['gt'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale'
    )
    return zip(dir_in_generator, dir_gt_generator)


def main():
    dataset_dir = '../dataset/data/dataset'
    dataset_size = len(os.listdir(dataset_dir)) // 2

    base_dir = './data/'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    train_percents = 80
    train_start = 0
    train_stop = int(dataset_size * (train_percents / 100))
    train_generator = gen_data(train_dir, dataset_dir, train_start, train_stop)

    validation_dir = os.path.join(base_dir, 'validation')
    validation_percents = 10
    validation_start = train_stop
    validation_stop = validation_start + int(dataset_size * (validation_percents / 100))
    validation_generator = gen_data(validation_dir, dataset_dir, validation_start, validation_stop)

    test_dir = os.path.join(base_dir, 'test')
    test_percents = 10
    test_start = validation_stop
    test_stop = dataset_size
    test_generator = gen_data(test_dir, dataset_dir, test_start, test_stop)

    model = unet()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=386,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=49
    )

    model.save('binarization.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_acc, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
