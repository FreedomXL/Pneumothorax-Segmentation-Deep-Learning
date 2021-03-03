
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Activation, add
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model


def conv_bn(input, filters, kernel_size, strides=(1, 1), activation='relu'):

    x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, padding='same', use_bias=False)(input)
    x = BN(axis=-1, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation)(x)
    return x


def MultiResBlock(input, U, alpha=1.67):

    W = alpha * U  # U: filters of corresponding U-Net layer

    # residual connection, adding 1*1 convolution
    shortcut = input
    shortcut = conv_bn(shortcut, int(W * 0.167)+int(W * 0.333)+int(W * 0.5), 1, activation=None)

    conv3x3 = conv_bn(input, int(W * 0.167), 3, activation='relu')
    conv5x5 = conv_bn(conv3x3, int(W * 0.333), 3, activation='relu')
    conv7x7 = conv_bn(conv5x5, int(W * 0.5), 3, activation='relu')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = BN(axis=-1)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BN(axis=-1)(out)

    return out


def ResPath(input, filters, length):

    shortcut = input
    shortcut = conv_bn(shortcut, filters, 1, activation=None)

    out = conv_bn(input, filters, 3, activation='relu')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BN(axis=-1)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv_bn(shortcut, filters, 1, activation=None)

        out = conv_bn(out, filters, 3, activation='relu')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BN(axis=-1)(out)

    return out


def MultiResUNet(input_shape):

    inputs = Input(input_shape)

    # ---------------------- Encoder ----------------------
    mresblock1 = MultiResBlock(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(mresblock1, 32, 4)

    mresblock2 = MultiResBlock(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(mresblock2, 64, 3)

    mresblock3 = MultiResBlock(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(mresblock3, 128, 2)

    mresblock4 = MultiResBlock(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(mresblock4, 256, 1)

    # bridge
    mresblock5 = MultiResBlock(pool4, 512)

    # ---------------------- Decoder ----------------------
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=-1)
    mresblock6 = MultiResBlock(up6, 256)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=-1)
    mresblock7 = MultiResBlock(up7, 128)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=-1)
    mresblock8 = MultiResBlock(up8, 64)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=-1)
    mresblock9 = MultiResBlock(up9, 32)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(mresblock9)

    model = Model(inputs=inputs, outputs=output, name='MultiResUNet')
    return model


if __name__ == '__main__':

    model = MultiResUNet(input_shape=(512, 512, 1))
    model.summary()
