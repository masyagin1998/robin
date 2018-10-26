import argparse

from keras.callbacks import ModelCheckpoint

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
    parser.add_argument('--in', type=str, default='./data/train/in/',
                        help=r'x size of image part (default: %(default)s)')
    parser.add_argument('--gt', type=str, default='./data/train/gt/',
                        help=r'x size of image part (default: %(default)s)')
    args = parser.parse_args()

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(2, 'data/train', 'image', 'label', data_gen_args, save_to_dir=None)
    model = unet()
    model_checkpoint = ModelCheckpoint('binarization.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
