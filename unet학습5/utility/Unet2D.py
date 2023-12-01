import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def SeperableConv(filter, input):
    depthwise = tf.keras.layers.Conv2D(
        input.shape[-1], 3, padding="same", groups=input.shape[-1]
    )(input)
    pointwise = tf.keras.layers.Conv2D(filter, 1, padding="same")(depthwise)
    return pointwise


def Block(filter, input):
    # conv1=tf.keras.layers.Conv3D(filter,3,padding="same")(input)
    conv1 = SeperableConv(filter, input)
    layerNorm1 = tf.keras.layers.LayerNormalization()(conv1)
    swishAct1 = tf.keras.layers.Activation("swish")(layerNorm1)

    # conv2=tf.keras.layers.Conv3D(filter,3,padding="same")(swishAct1)
    conv2 = SeperableConv(filter, swishAct1)
    layerNorm2 = tf.keras.layers.LayerNormalization()(conv2)
    swishAct2 = tf.keras.layers.Activation("swish")(layerNorm2)

    return swishAct2


def UnetModel(inputShape=(None, None, 12)):
    inputImage = tf.keras.Input(shape=inputShape)

    # 인코딩
    e1 = Block(32, inputImage)
    e1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e1)

    e2 = Block(64, e1Pooling)
    e2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e2)

    e3 = Block(128, e2Pooling)
    e3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e3)

    e4 = Block(256, e3Pooling)
    e4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e4)

    # 중간
    bottleNeck = Block(512, e4Pooling)

    d4UpSampling = tf.keras.layers.Conv2DTranspose(256, (2, 2), 2, padding="same")(
        bottleNeck
    )
    d4Concatenate = tf.keras.layers.Concatenate()([d4UpSampling, e4])
    d4 = Block(256, d4Concatenate)

    d3UpSampling = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2, padding="same")(d4)
    d3Concatenate = tf.keras.layers.Concatenate()([d3UpSampling, e3])
    d3 = Block(128, d3Concatenate)

    d2UpSampling = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2, padding="same")(d3)
    d2Concatenate = tf.keras.layers.Concatenate()([d2UpSampling, e2])
    d2 = Block(64, d2Concatenate)

    d1UpSampling = tf.keras.layers.Conv2DTranspose(32, (2, 2), 2, padding="same")(d2)
    d1Concatenate = tf.keras.layers.Concatenate()([d1UpSampling, e1])
    d1 = Block(32, d1Concatenate)

    outputImage = SeperableConv(8, d1)

    return tf.keras.Model(inputImage, outputImage)


if __name__ == "main":
    UnetModel().summary()
