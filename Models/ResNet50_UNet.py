
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


def bottleneck_identity_block(input, kernel_size, filters, pad):

    filters_1, filters_2, filters_3 = filters
    x_shortcut = input

    # stage 1
    x = conv_bn_relu(input, filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid')

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = conv_bn_relu(x, filters=filters_2, kernel_size=kernel_size, strides=(1, 1), padding='valid')

    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BN(axis=-1)(x)

    # stage 4
    x = add([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x


def bottleneck_convolutional_block(input, kernel_size, filters, strides, pad):

    filters_1, filters_2, filters_3 = filters
    x_shortcut = input

    # stage 1
    x = conv_bn_relu(input, filters=filters_1, kernel_size=(1, 1), strides=strides, padding='valid')

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = conv_bn_relu(x, filters=filters_2, kernel_size=kernel_size, strides=(1, 1), padding='valid')

    # stage 3
    x = conv_bn_relu(x, filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')

    # stage 4
    x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='valid')(x_shortcut)
    x_shortcut = BN(axis=-1)(x_shortcut)

    # stage 5
    x = add([x, x_shortcut])
    x = Activation(activation='relu')(x)
    return x


def ResNet50(inputs):  # 512 * 512 * 1

    # stage 1
    x = ZeroPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
    x = BN(axis=-1)(x)
    down_1 = Activation('relu')(x)  # 256 * 256 * 64
    x = ZeroPadding2D(padding=(1, 1))(down_1)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage 2
    x = bottleneck_convolutional_block(x, kernel_size=(3, 3), filters=[64, 64, 256], strides=1, pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1))
    down_2 = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1))  # 128 * 128 * 128

    # stage 3
    x = bottleneck_convolutional_block(down_2, kernel_size=(3, 3), filters=[128, 128, 512], strides=2, pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1))
    down_3 = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1))  # 64 * 64 * 256

    # stage 4
    x = bottleneck_convolutional_block(down_3, kernel_size=(3, 3), filters=[256, 256, 1024], strides=2, pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(1, 1))
    down_4 = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(1, 1))  # 32 * 32 * 512

    # stage 5
    x = bottleneck_convolutional_block(down_4, kernel_size=(3, 3), filters=[512, 512, 2048], strides=2, pad=(1, 1))
    x = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[512, 512, 2048], pad=(1, 1))
    down_5 = bottleneck_identity_block(x, kernel_size=(3, 3), filters=[512, 512, 2048], pad=(1, 1))  # 16 * 16 * 1024

    return down_1, down_2, down_3, down_4, down_5


def ResNet50_UNet(input_shape):

    inputs = Input(input_shape)

    # -------------------- Encoder -------------------
    # down_1: 256 * 256 * 64
    # down_2: 128 * 128 * 256
    # down_3: 64  * 64  * 512
    # down_4: 32  * 32  * 1024
    # down_5: 16  * 16  * 2048
    down_1, down_2, down_3, down_4, down_5 = ResNet50(inputs)

    # -------------------- Decoder -------------------
    up_3 = conv_block(concatenate([UpSampling2D(size=(2, 2))(down_4), down_3], axis=-1), 512)  # 64  * 64  * 512
    up_2 = conv_block(concatenate([UpSampling2D(size=(2, 2))(up_3), down_2], axis=-1), 256)    # 128 * 128 * 256
    up_1 = conv_block(concatenate([UpSampling2D(size=(2, 2))(up_2), down_1], axis=-1), 128)    # 256 * 256 * 128

    # one more upsampling in the end
    output = conv_block(UpSampling2D(size=(2, 2))(up_1), 64)  # 512 * 512 * 64
    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output, name='ResNet50_UNet')
    return model


if __name__ == '__main__':

    model = ResNet50_UNet(input_shape=(512, 512, 1))
    model.summary()
