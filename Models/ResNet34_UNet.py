
from keras.layers import Conv2D, concatenate, Input, Activation, add, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model


def conv_bn_relu(input, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    return x


def conv_block(input, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):

    x = conv_bn_relu(input, filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = conv_bn_relu(x, filters, kernel_size=kernel_size, strides=strides, padding=padding)

    return x


def basic_conv_block(input, filters, kernel_size, strides=(1, 1)):

    x = conv_bn_relu(input, filters, kernel_size=kernel_size, strides=strides, padding='same')

    x = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
    x = BN(axis=-1)(x)

    shortcut = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
    shortcut = BN(axis=-1)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def basic_identity_block(input, filters, kernel_size, strides=(1, 1)):

    shortcut = input

    x = conv_bn_relu(input, filters, kernel_size=kernel_size, strides=strides, padding='same')

    x = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
    x = BN(axis=-1)(x)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet34(inputs):  # 512 * 512 * 1

    # stage 1
    x = ZeroPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
    x = BN(axis=-1)(x)
    down_1 = Activation('relu')(x)  # 256 * 256 * 64
    x = ZeroPadding2D(padding=(1, 1))(down_1)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage 2
    x = basic_conv_block(x, filters=64, kernel_size=3)
    x = basic_identity_block(x, filters=64, kernel_size=3)
    down_2 = basic_identity_block(x, filters=64, kernel_size=3)  # 128 * 128 * 64

    # stage 3
    x = basic_conv_block(down_2, filters=128, kernel_size=3, strides=(2, 2))
    x = basic_identity_block(x, filters=128, kernel_size=3)
    x = basic_identity_block(x, filters=128, kernel_size=3)
    down_3 = basic_identity_block(x, filters=128, kernel_size=3)  # 64 * 64 * 128

    # stage 4
    x = basic_conv_block(down_3, filters=256, kernel_size=3, strides=(2, 2))
    x = basic_identity_block(x, filters=256, kernel_size=3)
    x = basic_identity_block(x, filters=256, kernel_size=3)
    x = basic_identity_block(x, filters=256, kernel_size=3)
    x = basic_identity_block(x, filters=256, kernel_size=3)
    down_4 = basic_identity_block(x, filters=256, kernel_size=3)  # 32 * 32 * 256

    # stage 5
    x = basic_conv_block(down_4, filters=512, kernel_size=3, strides=(2, 2))
    x = basic_identity_block(x, filters=512, kernel_size=3)
    down_5 = basic_identity_block(x, filters=512, kernel_size=3)  # 16 * 16 * 512

    return down_1, down_2, down_3, down_4, down_5


def ResNet34_UNet(input_shape):

    inputs = Input(input_shape)

    # -------------------- Encoder -------------------
    # down_1: 256 * 256 * 64
    # down_2: 128 * 128 * 64
    # down_3: 64  * 64  * 128
    # down_4: 32  * 32  * 256
    # down_5: 16  * 16  * 512
    down_1, down_2, down_3, down_4, down_5 = ResNet34(inputs)

    # -------------------- Decoder -------------------
    # experiments show that applying 3 decoder layers is better than 4
    up_3 = conv_block(concatenate([UpSampling2D(size=(2, 2))(down_4), down_3], axis=-1), 128)  # 64  * 64  * 128
    up_2 = conv_block(concatenate([UpSampling2D(size=(2, 2))(up_3), down_2], axis=-1), 64)     # 128 * 128 * 64
    up_1 = conv_block(concatenate([UpSampling2D(size=(2, 2))(up_2), down_1], axis=-1), 32)     # 256 * 256 * 32

    # one more upsampling in the end
    output = conv_block(UpSampling2D(size=(2, 2))(up_1), 16)  # 512 * 512 * 16
    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output, name='ResNet34_UNet')
    return model


if __name__ == '__main__':

    model = ResNet34_UNet(input_shape=(512, 512, 1))
    model.summary()
