
from keras.models import Model
from keras.layers import Conv2D, Activation, Input, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization as BN


def conv_bn_relu(input, filters, kernel_size=3):

    conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(input)
    conv = BN()(conv)
    conv = Activation(activation='relu')(conv)

    return conv


def conv_block(input, filters, kernel_size=3):

    conv = conv_bn_relu(input, filters, kernel_size=kernel_size)
    conv = conv_bn_relu(conv, filters, kernel_size=kernel_size)

    return conv


def UNet3Plus(input_shape):

    inputs = Input(input_shape)

    # ---------------------- Encoder ----------------------
    conv0 = conv_block(inputs, 32)
    conv1 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv0), 64)
    conv2 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1), 128)
    conv3 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2), 256)
    conv4 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv3), 512)

    # ---------------------- Decoder ----------------------
    # bilinear interpolation
    deconv4 = conv4  # 32 * 32 * 512

    # deconv3: 64 * 64 * 160
    deconv3_0 = conv_bn_relu(MaxPooling2D(pool_size=(8, 8))(conv0), 32)  # 512 * 512 * 32  => 64 * 64 * 32
    deconv3_1 = conv_bn_relu(MaxPooling2D(pool_size=(4, 4))(conv1), 32)  # 256 * 256 * 64  => 64 * 64 * 32
    deconv3_2 = conv_bn_relu(MaxPooling2D(pool_size=(2, 2))(conv2), 32)  # 128 * 128 * 128 => 64 * 64 * 32
    deconv3_3 = conv_bn_relu(conv3, 32)                                  # 64  * 64  * 256 => 64 * 64 * 32
    deconv3_4 = conv_bn_relu(UpSampling2D(size=(2, 2), interpolation='bilinear')(deconv4), 32)  # 32 * 32 * 512 => 64 * 64 * 32
    deconv3 = conv_bn_relu(concatenate([deconv3_0, deconv3_1, deconv3_2, deconv3_3, deconv3_4], axis=-1), 32*5)

    # deconv2: 128 * 128 * 160
    deconv2_0 = conv_bn_relu(MaxPooling2D(pool_size=(4, 4))(conv0), 32)  # 512 * 512 * 32  => 128 * 128 * 32
    deconv2_1 = conv_bn_relu(MaxPooling2D(pool_size=(2, 2))(conv1), 32)  # 256 * 256 * 64  => 128 * 128 * 32
    deconv2_2 = conv_bn_relu(conv2, 32)                                  # 128 * 128 * 128 => 128 * 128 * 32
    deconv2_3 = conv_bn_relu(UpSampling2D(size=(2, 2), interpolation='bilinear')(deconv3), 32)  # 64 * 64 * 160 => 128 * 128 * 32
    deconv2_4 = conv_bn_relu(UpSampling2D(size=(4, 4), interpolation='bilinear')(deconv4), 32)  # 32 * 32 * 512 => 128 * 128 * 32
    deconv2 = conv_bn_relu(concatenate([deconv2_0, deconv2_1, deconv2_2, deconv2_3, deconv2_4], axis=-1), 32*5)

    # deconv1: 256 * 256 * 160
    deconv1_0 = conv_bn_relu(MaxPooling2D(pool_size=(2, 2))(conv0), 32)  # 512 * 512 * 32 => 256 * 256 * 32
    deconv1_1 = conv_bn_relu(conv1, 32)                                  # 256 * 256 * 64 => 256 * 256 * 32
    deconv1_2 = conv_bn_relu(UpSampling2D(size=(2, 2), interpolation='bilinear')(deconv2), 32)  # 128 * 128 * 160 => 256 * 256 * 32
    deconv1_3 = conv_bn_relu(UpSampling2D(size=(4, 4), interpolation='bilinear')(deconv3), 32)  # 64  * 64  * 160 => 256 * 256 * 32
    deconv1_4 = conv_bn_relu(UpSampling2D(size=(8, 8), interpolation='bilinear')(deconv4), 32)  # 32  * 32  * 512 => 256 * 256 * 32
    deconv1 = conv_bn_relu(concatenate([deconv1_0, deconv1_1, deconv1_2, deconv1_3, deconv1_4], axis=-1), 32*5)

    # deconv0: 512 * 512 * 160
    deconv0_0 = conv_bn_relu(conv0, 32)  # 512 * 512 * 32 => 512 * 512 * 32
    deconv0_1 = conv_bn_relu(UpSampling2D(size=(2, 2), interpolation='bilinear')(deconv1), 32)    # 256 * 256 * 160 => 512 * 512 * 32
    deconv0_2 = conv_bn_relu(UpSampling2D(size=(4, 4), interpolation='bilinear')(deconv2), 32)    # 128 * 128 * 160 => 512 * 512 * 32
    deconv0_3 = conv_bn_relu(UpSampling2D(size=(8, 8), interpolation='bilinear')(deconv3), 32)    # 64  * 64  * 160 => 512 * 512 * 32
    deconv0_4 = conv_bn_relu(UpSampling2D(size=(16, 16), interpolation='bilinear')(deconv4), 32)  # 32  * 32  * 512 => 512 * 512 * 32
    deconv0 = conv_bn_relu(concatenate([deconv0_0, deconv0_1, deconv0_2, deconv0_3, deconv0_4], axis=-1), 32*5)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(deconv0)

    model = Model(inputs=inputs, outputs=output, name='UNet3Plus')
    return model


if __name__ == '__main__':

    model = UNet3Plus(input_shape=(512, 512, 1))
    model.summary()
