
from keras.layers import Conv2D, Concatenate, Input, Lambda, ZeroPadding2D, AveragePooling2D, Activation, Add, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
import tensorflow as tf
import keras.backend as K


def identity_block(x, kernel_size, filters, dilation, pad):

    filters_1, filters_2, filters_3 = filters
    x_shortcut = x

    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = Conv2D(filters=filters_2, kernel_size=kernel_size, strides=(1, 1), dilation_rate=dilation)(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BN(axis=-1)(x)

    # stage 4
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


def convolutional_block(x, kernel_size, filters, strides, dilation, pad):

    filters_1, filters_2, filters_3 = filters
    x_shortcut = x

    # stage 1
    x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=strides, padding='valid')(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # stage 2
    x = ZeroPadding2D(padding=pad)(x)
    x = Conv2D(filters=filters_2, kernel_size=kernel_size, strides=(1, 1), dilation_rate=dilation)(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # stage 3
    x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # stage 4
    x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='valid')(x_shortcut)
    x_shortcut = BN(axis=-1)(x_shortcut)

    # stage 5
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


# 512 * 512 * 1 => 64 * 64 * 1024
def ResNet50(inputs):

    # stage 1
    x = ZeroPadding2D(padding=(1, 1))(inputs)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x_stage_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage 2
    x = convolutional_block(x_stage_1, kernel_size=(3, 3), filters=[32, 32, 128], strides=1, pad=(1, 1), dilation=1)
    x = identity_block(x, kernel_size=(3, 3), filters=[32, 32, 128], pad=(1, 1), dilation=1)
    x_stage_2 = identity_block(x, kernel_size=(3, 3), filters=[32, 32, 128], pad=(1, 1), dilation=1)

    # stage 3
    x = convolutional_block(x_stage_2, kernel_size=(3, 3), filters=[64, 64, 256], strides=2, pad=(1, 1), dilation=1)
    x = identity_block(x, kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
    x = identity_block(x, kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
    x_stage_3 = identity_block(x, kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)

    # stage 4
    x = convolutional_block(x_stage_3, kernel_size=(3, 3), filters=[128, 128, 512], strides=1, pad=(2, 2), dilation=2)
    x = identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(2, 2), dilation=2)
    x = identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(2, 2), dilation=2)
    x = identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(2, 2), dilation=2)
    x = identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(2, 2), dilation=2)
    x_stage_4 = identity_block(x, kernel_size=(3, 3), filters=[128, 128, 512], pad=(2, 2), dilation=2)

    # stage 5
    x = convolutional_block(x_stage_4, kernel_size=(3, 3), filters=[256, 256, 1024], strides=1, pad=(4, 4), dilation=4)
    x = identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(4, 4), dilation=4)
    x_stage_5 = identity_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], pad=(4, 4), dilation=4)

    return x_stage_5


def resize_image(args):
    '''
        resize the feature map using bilinear interpolation
        :args: [x, standard]
        :return: resized x whose feature map shape is the same as standard
    '''
    x = args[0]
    standard = args[1]
    shape = K.int_shape(standard)
    return tf.compat.v1.image.resize_images(x, (shape[1], shape[2]), align_corners=True)


def PSPNet(input_shape):

    inputs = Input(shape=input_shape)

    res_features = ResNet50(inputs)  # 64 * 64 * 1024

    # pyramid pooling
    # pool_size = 60, 30, 20, 10, padding = valid in AveragePooling2D
    x_c1 = AveragePooling2D(pool_size=60, strides=60, name='ave_c1')(res_features)              # 1 * 1 * 1024
    x_c1 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', name='conv_c1')(x_c1)  # 1 * 1 * 256
    x_c1 = BN(axis=-1)(x_c1)
    x_c1 = Activation('relu')(x_c1)
    x_c1 = Lambda(resize_image)([x_c1, res_features])  # 64 * 64 * 256

    x_c2 = AveragePooling2D(pool_size=30, strides=30, name='ave_c2')(res_features)              # 2 * 2 * 1024
    x_c2 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', name='conv_c2')(x_c2)  # 2 * 2 * 256
    x_c2 = BN(axis=-1)(x_c2)
    x_c2 = Activation('relu')(x_c2)
    x_c2 = Lambda(resize_image)([x_c2, res_features])  # 64 * 64 * 256

    x_c3 = AveragePooling2D(pool_size=20, strides=20, name='ave_c3')(res_features)              # 3 * 3 * 1024
    x_c3 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', name='conv_c3')(x_c3)  # 3 * 3 * 256
    x_c3 = BN(axis=-1)(x_c3)
    x_c3 = Activation('relu')(x_c3)
    x_c3 = Lambda(resize_image)([x_c3, res_features])  # 64 * 64 * 256

    x_c4 = AveragePooling2D(pool_size=10, strides=10, name='ave_c4')(res_features)              # 6 * 6 * 1024
    x_c4 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', name='conv_c4')(x_c4)  # 6 * 6 * 256
    x_c4 = BN(axis=-1)(x_c4)
    x_c4 = Activation('relu')(x_c4)
    x_c4 = Lambda(resize_image)([x_c4, res_features])  # 64 * 64 * 256

    # 64 * 64 * (out_channel * 2) => 64 * 64 * 2048
    x = Concatenate(axis=-1, name='concat')([x_c1, x_c2, x_c3, x_c4, res_features])

    # 64 * 64 * (out_channel // 4) => 64 * 64 * 256
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='sum_conv_1')(x)
    x = BN(axis=-1)(x)
    x = Activation('relu')(x)

    # 64 * 64 * 256 => 512 * 512 * 256
    x = Lambda(resize_image)([x, inputs])

    outputs = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='sum_conv_2')(x)  # 512 * 512 * 1

    model = Model(inputs=inputs, outputs=outputs, name='PSPNet')
    return model


if __name__ == '__main__':

    model = PSPNet(input_shape=(512, 512, 1))
    model.summary()
