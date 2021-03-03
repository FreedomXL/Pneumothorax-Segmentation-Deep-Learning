
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, UpSampling2D
from keras.layers.normalization import BatchNormalization as BN


def conv_bn_relu(input, filters, kernel_size=3, dilation=False):

    if dilation == True:
        # from experimental results, dilation rate = 3 is better than 2
        dilation_rate = (3, 3)
    else:
        dilation_rate = (1, 1)

    conv = Conv2D(filters, (kernel_size, kernel_size), dilation_rate=dilation_rate, padding='same')(input)
    conv = BN(axis=-1)(conv)
    conv = Activation('relu')(conv)

    return conv


def conv_block(input, filters, kernel_size=3, dilation=False):

    conv = conv_bn_relu(input, filters, kernel_size=kernel_size, dilation=dilation)
    conv = conv_bn_relu(conv, filters, kernel_size=kernel_size, dilation=dilation)

    return conv


def dilated_UNet(input_shape):

    inputs = Input(input_shape)

    # ---------------------- Encoder ----------------------
    conv1 = conv_block(inputs, 32)
    conv2 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1), 64, dilation=True)
    conv3 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2), 128, dilation=True)
    conv4 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv3), 256, dilation=True)
    # dilated convolution is not used in bridge
    conv5 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv4), 512)

    # ---------------------- Decoder ----------------------
    deconv4 = conv_block(concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1), 256)
    deconv3 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv4), conv3], axis=-1), 128)
    deconv2 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv3), conv2], axis=-1), 64)
    deconv1 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv2), conv1], axis=-1), 32)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(deconv1)

    model = Model(inputs=inputs, outputs=output, name='dilated_UNet')
    return model


if __name__ == '__main__':

    model = dilated_UNet(input_shape=(512, 512, 1))
    model.summary()
