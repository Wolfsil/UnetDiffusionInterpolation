import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Warping(tf.keras.layers.Layer):
    def __init__(self):
        super(Warping, self).__init__()

    def call(self, inputs):  # 모델의 input과 output 계산방식 선언
        return tfa.image.dense_image_warp(inputs[0], inputs[1])


def Block(filter, input, kernelSize):
    conv1 = tf.keras.layers.SeparableConv2D(
        filter, kernel_size=kernelSize, padding="same"
    )(input)

    layerNorm1 = tf.keras.layers.LayerNormalization()(conv1)

    swishAct1 = tf.keras.layers.Activation("swish")(layerNorm1)

    convW1 = tf.keras.layers.SeparableConv2D(2, kernel_size=kernelSize, padding="same")(
        input
    )

    input2 = Warping()([swishAct1, convW1])

    conv2 = tf.keras.layers.SeparableConv2D(
        filter, kernel_size=kernelSize, padding="same"
    )(input2)

    layerNorm2 = tf.keras.layers.LayerNormalization()(conv2)

    swishAct2 = tf.keras.layers.Activation("swish")(layerNorm2)

    convW2 = tf.keras.layers.SeparableConv2D(2, kernel_size=kernelSize, padding="same")(
        input2
    )

    output = Warping()([swishAct2, convW2])

    return output


def UnetModel(inputShape=(128, 128, 4)):
    primaryImage = tf.keras.Input(shape=inputShape)
    secondaryImage = tf.keras.Input(shape=inputShape)

    primaryImage2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(primaryImage)
    primaryImage3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(primaryImage2)
    primaryImage4 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(primaryImage3)

    secondaryImage2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(secondaryImage)
    secondaryImage3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(
        secondaryImage2
    )
    secondaryImage4 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(
        secondaryImage3
    )

    # 인코더

    e1 = Block(32, tf.keras.layers.Concatenate()([primaryImage, secondaryImage]), 17)

    e1Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(e1)

    e2 = Block(
        64,
        tf.keras.layers.Concatenate()([primaryImage2, e1Pooling, secondaryImage2]),
        9,
    )

    e2Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(e2)

    e3 = Block(
        128,
        tf.keras.layers.Concatenate()([primaryImage3, e2Pooling, secondaryImage3]),
        5,
    )

    e3Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(e3)

    e4 = Block(
        256,
        tf.keras.layers.Concatenate()([primaryImage4, e3Pooling, secondaryImage4]),
        3,
    )

    e4Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(e4)

    # 보틀넥
    bottleNeck = Block(512, e4Pooling, 3)

    # 디코더
    d4UpSampling = tf.keras.layers.UpSampling2D(size=(2, 2))(bottleNeck)
    d4Transpose = tf.keras.layers.SeparableConv2D(256, kernel_size=3, padding="same")(
        d4UpSampling
    )
    d4Concatenate = tf.keras.layers.Concatenate()(
        [primaryImage4, e4, d4Transpose, secondaryImage4]
    )
    d4 = Block(256, d4Concatenate, 3)

    d3UpSampling = tf.keras.layers.UpSampling2D(size=(2, 2))(d4)
    d3Transpose = tf.keras.layers.SeparableConv2D(128, kernel_size=3, padding="same")(
        d3UpSampling
    )
    d3Concatenate = tf.keras.layers.Concatenate()(
        [primaryImage3, e3, d3Transpose, secondaryImage3]
    )
    d3 = Block(128, d3Concatenate, 5)

    d2UpSampling = tf.keras.layers.UpSampling2D(size=(2, 2))(d3)
    d2Transpose = tf.keras.layers.SeparableConv2D(64, kernel_size=3, padding="same")(
        d2UpSampling
    )
    d2Concatenate = tf.keras.layers.Concatenate()(
        [primaryImage2, e2, d2Transpose, secondaryImage2]
    )
    d2 = Block(64, d2Concatenate, 9)

    d1UpSampling = tf.keras.layers.UpSampling2D(size=(2, 2))(d2)
    d1Transpose = tf.keras.layers.SeparableConv2D(32, kernel_size=3, padding="same")(
        d1UpSampling
    )
    d1Concatenate = tf.keras.layers.Concatenate()(
        [primaryImage, e1, d1Transpose, secondaryImage]
    )
    d1 = Block(32, d1Concatenate, 17)

    outputImage = tf.keras.layers.SeparableConv2D(4, kernel_size=1, padding="same")(d1)

    return tf.keras.Model([primaryImage, secondaryImage], outputImage)


if __name__ == "main":
    UnetModel().summary()
