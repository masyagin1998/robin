import keras.backend as K
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization,
                          Activation, SpatialDropout2D)
from keras.models import Model


def downsampling(filters, kernel_size, inputs):
    """Create downsampling layer."""
    conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #conv = BatchNormalization(momentum=0.9)(conv)
    conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    #conv = BatchNormalization(momentum=0.9)(conv)
    drop = Dropout(0.1)(conv)

    return drop


def upsampling(filters, kernels_size_first, kernels_size_second, inputs, concats):
    """Create upsampling layer."""
    up = Conv2D(filters, kernels_size_first, activation='relu',
                padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(inputs))
    merge = concatenate([concats, up], axis=3)
    conv = downsampling(filters, kernels_size_second, merge)

    return conv


def unet():
    """Create U-net model."""
    inputs = Input((128, 128, 1))

    # Contracting/downsampling path.
    down1 = downsampling(64, 3, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)

    down2 = downsampling(128, 3, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)

    down3 = downsampling(256, 3, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)

    down4 = downsampling(512, 3, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)

    # Bottleneck.
    bottleneck = downsampling(1024, 4, pool4)

    # Expanding/upsampling path.
    up6 = upsampling(512, 2, 3, bottleneck, down4)

    up7 = upsampling(256, 2, 3, up6, down3)

    up8 = upsampling(128, 2, 3, up7, down2)

    up9 = upsampling(64, 2, 3, up8, down1)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    return Model(input=inputs, output=conv10)

