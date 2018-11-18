#!/usr/bin/python3

import argparse
import time
from shutil import (rmtree, copy2)

import imageio
from alt_model_checkpoint import AltModelCheckpoint
from keras import backend as K
from keras.callbacks import (EarlyStopping, TensorBoard, Callback)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from model.unet import unet
from utils.augmentations import random_effect_img
from utils.img_processing import *


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
        copy2(src, dst)
        src = os.path.join(dataset_dname, 'gt', fname.replace('in', 'gt'))
        dst = os.path.join(dir_gt, fname)
        copy2(src, dst)

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
        if augmentate:
            img_in, img_gt = random_effect_img(img_in, img_gt)
        img_in = normalize_in(img_in)
        img_gt = normalize_gt(img_gt)
        yield (img_in, img_gt)


class Visualisation(Callback):
    """Custom Keras callback for visualising training through GIFs."""

    def __init__(self, dir_name: str = 'visualisation', batchsize: int = 20,
                 monitor: str = 'val_loss', save_best_epochs_only: bool = False, mode: str = 'min'):
        super(Visualisation, self).__init__()
        self.dir_name = dir_name
        self.batchsize = batchsize
        self.epoch_number = 0
        self.fnames = os.listdir(self.dir_name)
        for fname in self.fnames:
            mkdir_s(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))
        self.monitor = monitor
        self.save_best_epochs_only = save_best_epochs_only
        self.mode = mode
        self.curr_metric = None

    def on_train_end(self, logs=None):
        for fname in self.fnames:
            frames = []
            for frame_name in sorted(os.listdir(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))):
                frames.append(imageio.imread(os.path.join(self.dir_name,
                                                          fname[:fname.rfind('.')] + '_frames',
                                                          frame_name)))
            imageio.mimsave(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '.gif'),
                            frames, format='GIF', duration=0.5)
            # rmtree(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))

    def on_epoch_end(self, epoch, logs):
        self.epoch_number += 1
        if (not self.save_best_epochs_only) or \
                ((self.curr_metric is None) or
                 (self.mode == 'min' and logs[self.monitor] < self.curr_metric) or
                 (self.mode == 'max' and logs[self.monitor] > self.curr_metric)):
            self.curr_metric = logs[self.monitor]
            for fname in self.fnames:
                img = cv2.imread(os.path.join(self.dir_name, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = binarize_img(img, self.model, self.batchsize)
                cv2.imwrite(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames',
                                         str(self.epoch_number) + '_out.png'), img)


def create_callbacks(model, original_model, args):
    """Create Keras callbacks for training."""
    callbacks = []

    # Model checkpoint.
    if args.gpus == 1:
        model_checkpoint = AltModelCheckpoint(args.weights if args.debug == ''
                                              else os.path.join(args.debug, 'weights',
                                                                'weights-improvement-{epoch:02d}.hdf5'),
                                              model, monitor='val_dice_coef', mode='max', verbose=1,
                                              save_best_only=True, save_weights_only=True)
    else:
        model_checkpoint = AltModelCheckpoint(args.weights if args.debug == ''
                                              else os.path.join(args.debug, 'weights',
                                                                'weights-improvement-{epoch:02d}.hdf5'),
                                              original_model, monitor='val_dice_coef', mode='max', verbose=1,
                                              save_best_only=True, save_weights_only=True)
    callbacks.append(model_checkpoint)

    # Early stopping.
    model_early_stopping = EarlyStopping(monitor='val_dice_coef', min_delta=0.001, patience=8, verbose=1, mode='max')
    callbacks.append(model_early_stopping)

    # Tensorboard logs.
    if args.debug != '':
        mkdir_s(args.debug)
        mkdir_s(os.path.join(args.debug, 'weights'))
        mkdir_s(os.path.join(args.debug, 'logs'))
        model_tensorboard = TensorBoard(log_dir=os.path.join(args.debug, 'logs'),
                                        histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(model_tensorboard)

    # Training visualisation.
    if args.vis != '':
        model_visualisation = Visualisation(dir_name=args.vis, batchsize=args.batchsize, monitor='val_dice_coef',
                                            save_best_epochs_only=True, mode='max')
        callbacks.append(model_visualisation)

    return callbacks


def dice_coef(y_true, y_pred):
    """Count Sorensen-Dice coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    """Count loss of Sorensen-Dice coefficient for output and ground-truth image."""
    return 1 - dice_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    """Count Jaccard coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    """Count loss of Jaccard coefficient for output and ground-truth image."""
    return 1 - jacard_coef(y_true, y_pred)


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
    parser.add_argument('--vis', type=str, default='',
                        help=r'directory with images for training visualisation')
    return parser.parse_args()


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
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss,
                      metrics=[dice_coef, jacard_coef, 'accuracy'])
    else:
        model = multi_gpu_model(original_model, gpus=args.gpus)
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss,
                      metrics=[dice_coef, jacard_coef, 'accuracy'])

    callbacks = create_callbacks(model, original_model, args)

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
    print()
    print('total:')
    print('test_loss:       {0:.4f}'.format(metrics[0]))
    print('test_dice_coef:  {0:.4f}'.format(metrics[1]))
    print('test_jacar_coef: {0:.4f}'.format(metrics[2]))
    print('test_accuracy:   {0:.4f}'.format(metrics[3]))

    rmtree(tmp)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
