import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Warping(tf.keras.layers.Layer):
    def __init__(self):
        super(Warping, self).__init__()

    def call(self, inputs):  # 모델의 input과 output 계산방식 선언
        return tfa.image.dense_image_warp(inputs[0], inputs[1])


def SeperableConv(filter, input):
    depthwise = tf.keras.layers.Conv2D(
        input.shape[-1], 3, padding="same", groups=input.shape[-1]
    )(input)

    pointwise = tf.keras.layers.Conv2D(filter, 1, padding="same")(depthwise)

    return pointwise


def Block(filter, input):
    conv1 = SeperableConv(filter, input)

    layerNorm1 = tf.keras.layers.LayerNormalization()(conv1)

    swishAct1 = tf.keras.layers.Activation("swish")(layerNorm1)

    conv2 = SeperableConv(filter, swishAct1)

    layerNorm2 = tf.keras.layers.LayerNormalization()(conv2)

    swishAct2 = tf.keras.layers.Activation("swish")(layerNorm2)

    return swishAct2


def UnetModel(inputShape=(None, None, 4), stepShape=(None, None, 1)):
    primaryImage = tf.keras.Input(shape=inputShape)
    secondaryImage = tf.keras.Input(shape=inputShape)

    # 프라이머리 인코더

    pe1 = Block(32, primaryImage)

    pe1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe1)

    pe2 = Block(64, pe1Pooling)

    pe2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe2)

    pe3 = Block(128, pe2Pooling)

    pe3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe3)

    pe4 = Block(256, pe3Pooling)

    pe4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe4)

    # 세컨더리 인코더

    se1 = Block(32, secondaryImage)

    se1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se1)

    se2 = Block(64, se1Pooling)

    se2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se2)

    se3 = Block(128, se2Pooling)

    se3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se3)

    se4 = Block(256, se3Pooling)

    se4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se4)

    ###

    # 보틀넥
    mainbottleNeck = Block(512, tf.keras.layers.Concatenate()([pe4Pooling, se4Pooling]))

    # 디코더
    maind4UpSampling = tf.keras.layers.Conv2DTranspose(256, (2, 2), 2, padding="same")(
        mainbottleNeck
    )
    maind4Concatenate = tf.keras.layers.Concatenate()([pe4, maind4UpSampling, se4])
    maind4 = Block(256, maind4Concatenate)
    maind4WarpMap = Block(2, maind4Concatenate)
    maind4Warp = Warping()([maind4, maind4WarpMap])

    maind3UpSampling = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2, padding="same")(
        maind4Warp
    )
    maind3Concatenate = tf.keras.layers.Concatenate()([pe3, maind3UpSampling, se3])
    maind3 = Block(128, maind3Concatenate)
    maind3WarpMap = Block(2, maind3Concatenate)
    maind3Warp = Warping()([maind3, maind3WarpMap])

    maind2UpSampling = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2, padding="same")(
        maind3Warp
    )
    maind2Concatenate = tf.keras.layers.Concatenate()([pe2, maind2UpSampling, se2])
    maind2 = Block(64, maind2Concatenate)
    maind2WarpMap = Block(2, maind2Concatenate)
    maind2Warp = Warping()([maind2, maind2WarpMap])

    maind1UpSampling = tf.keras.layers.Conv2DTranspose(32, (2, 2), 2, padding="same")(
        maind2Warp
    )
    maind1Concatenate = tf.keras.layers.Concatenate()([pe1, maind1UpSampling, se1])
    maind1 = Block(32, maind1Concatenate)
    maind1WarpMap = Block(2, maind1Concatenate)
    maind1Warp = Warping()([maind1, maind1WarpMap])

    outputImage = SeperableConv(4, maind1Warp)

    return tf.keras.Model([primaryImage, secondaryImage], outputImage)


if __name__ == "main":
    UnetModel().summary()
