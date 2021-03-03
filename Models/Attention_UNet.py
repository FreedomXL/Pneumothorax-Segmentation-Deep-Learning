
from keras.layers import Input, Conv2D, Activation, add, UpSampling2D, multiply, MaxPooling2D, concatenate, Lambda
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
import keras.backend as K


def conv_bn_relu(input, filters, kernel_size=3):

    conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(input)
    conv = BN(axis=-1)(conv)
    conv = Activation('relu')(conv)

    return conv


def res_block(input, filters, kernel_size=3):

    conv = conv_bn_relu(input, filters, kernel_size)
    conv = conv_bn_relu(conv, filters, kernel_size)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same')(input)
    shortcut = BN(axis=-1)(shortcut)

    res_path = add([shortcut, conv])
    return res_path


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1),
                         arguments={'repnum': rep})(tensor)


def attention_block(x, gating, inter_channel):
    '''
        inter_channel: intermediate channel
    '''
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv2D(inter_channel, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(gating)
    upsample_g = UpSampling2D(size=(shape_theta_x[1]//shape_g[1], shape_theta_x[2]//shape_g[2]))(phi_g)

    relu_xg = Activation('relu')(add([theta_x, upsample_g]))

    psi = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu_xg)
    sigmoid_psi = Activation('sigmoid')(psi)
    shape_sigmoid_psi = K.int_shape(sigmoid_psi)

    upsample_psi = UpSampling2D(size=(shape_x[1]//shape_sigmoid_psi[1], shape_x[2]//shape_sigmoid_psi[2]))(sigmoid_psi)
    upsample_psi = expend_as(upsample_psi, shape_x[3])

    out = multiply([upsample_psi, x])
    out = Conv2D(shape_x[3], kernel_size=(1, 1), padding='same')(out)
    out = BN(axis=-1)(out)

    return out


def Attention_UNet(input_shape):

    inputs = Input(input_shape)

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_1 = res_block(inputs, 32, 3)
    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

    # DownRes 2
    conv_2 = res_block(pool_1, 64, 3)
    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

    # DownRes 3
    conv_3 = res_block(pool_2, 128, 3)
    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

    # DownRes 4
    conv_4 = res_block(pool_3, 256, 3)
    pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

    # DownRes 5, convolution only
    conv_5 = res_block(pool_4, 512, 3)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_4 = conv_bn_relu(conv_5, 256, kernel_size=1)
    attention_4 = attention_block(conv_4, gating_4, 256)
    up_4 = UpSampling2D(size=(2, 2))(conv_5)
    up_4 = concatenate([up_4, attention_4], axis=-1)
    up_conv_4 = res_block(up_4, 256, 3)

    # UpRes 7
    gating_3 = conv_bn_relu(up_conv_4, 128, kernel_size=1)
    attention_3 = attention_block(conv_3, gating_3, 128)
    up_3 = UpSampling2D(size=(2, 2))(up_conv_4)
    up_3 = concatenate([up_3, attention_3], axis=-1)
    up_conv_3 = res_block(up_3, 128, 3)

    # UpRes 8
    gating_2 = conv_bn_relu(up_conv_3, 64, kernel_size=1)
    attention_2 = attention_block(conv_2, gating_2, 64)
    up_2 = UpSampling2D(size=(2, 2))(up_conv_3)
    up_2 = concatenate([up_2, attention_2], axis=-1)
    up_conv_2 = res_block(up_2, 64, 3)

    # UpRes 9
    gating_1 = conv_bn_relu(up_conv_2, 32, kernel_size=1)
    attention_1 = attention_block(conv_1, gating_1, 32)
    up_1 = UpSampling2D(size=(2, 2))(up_conv_2)
    up_1 = concatenate([up_1, attention_1], axis=-1)
    up_conv_1 = res_block(up_1, 32, 3)

    # 1*1 convolutional layers, valid padding
    output = Conv2D(1, kernel_size=(1,1))(up_conv_1)
    output = BN(axis=-1)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=inputs, outputs=output, name='Attention_UNet')
    return model


if __name__ == '__main__':

    model = Attention_UNet(input_shape=(512, 512, 1))
    model.summary()
