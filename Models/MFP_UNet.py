
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


def MFP_UNet(input_shape):

    inputs = Input(input_shape)

    # ---------------------- Encoder ----------------------
    conv1 = conv_block(inputs, 32, dilation=True)
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

    # ------------------ Feature Pyramid ------------------
    deconv1 = conv_bn_relu(deconv1, 32)  # 512 * 512 * 32  => 512 * 512 * 32
    deconv2 = conv_bn_relu(deconv2, 32)  # 256 * 256 * 64  => 256 * 256 * 32
    deconv3 = conv_bn_relu(deconv3, 32)  # 128 * 128 * 128 => 128 * 128 * 32
    deconv4 = conv_bn_relu(deconv4, 32)  # 64  * 64  * 256 => 64  * 64  * 32
    conv5   = conv_bn_relu(conv5, 32)    # 32  * 32  * 512 => 32  * 32  * 32

    deconv2 = UpSampling2D(size=(2, 2))(deconv2)  # 256 * 256 * 32 => 512 * 512 * 32
    deconv3 = UpSampling2D(size=(4, 4))(deconv3)  # 128 * 128 * 32 => 512 * 512 * 32
    deconv4 = UpSampling2D(size=(8, 8))(deconv4)  # 64  * 64  * 32 => 512 * 512 * 32
    conv5   = UpSampling2D(size=(16, 16))(conv5)  # 32  * 32  * 32 => 512 * 512 * 32

    output = concatenate([deconv1, deconv2, deconv3, deconv4, conv5], axis=-1)  # 512 * 512 * 160
    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output, name='MFP_UNet')
    return model


if __name__ == '__main__':

    model = MFP_UNet(input_shape=(512, 512, 1))
    model.summary()
