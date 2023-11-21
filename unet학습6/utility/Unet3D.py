import tensorflow as tf


def SeperableConv(filter, input):
    depthwise = tf.keras.layers.Conv3D(
        input.shape[-1], 3, padding="same", groups=input.shape[-1]
    )(input)
    pointwise = tf.keras.layers.Conv3D(filter, 1, padding="same")(depthwise)
    return pointwise


def SeperableConvEnd(filter, input):
    depthwise = tf.keras.layers.Conv3D(
        input.shape[-1], 3, padding="same", groups=input.shape[-1]
    )(input)
    pointwise = tf.keras.layers.Conv3D(filter, 1, padding="same", dtype=tf.float32)(
        depthwise
    )
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


def UnetModel(inputShape=(7, 64, 64, 4)):
    inputImage = tf.keras.Input(shape=inputShape)

    # 생성 unet
    # 인코딩
    e0 = Block(16, inputImage)
    e0Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e0)

    e1 = Block(32, e0Pooling)
    e1Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e1)

    e2 = Block(64, e1Pooling)
    e2Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e2)

    e3 = Block(128, e2Pooling)
    e3Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e3)

    e4 = Block(256, e3Pooling)
    e4Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e4)

    # 중간
    bottleNeck = Block(512, e4Pooling)

    d4UpSampling = tf.keras.layers.Conv3DTranspose(
        256, (1, 2, 2), (1, 2, 2), padding="same"
    )(bottleNeck)
    d4Concatenate = tf.keras.layers.Concatenate()([d4UpSampling, e4])
    d4 = Block(256, d4Concatenate)

    d3UpSampling = tf.keras.layers.Conv3DTranspose(
        128, (1, 2, 2), (1, 2, 2), padding="same"
    )(d4)
    d3Concatenate = tf.keras.layers.Concatenate()([d3UpSampling, e3])
    d3 = Block(128, d3Concatenate)

    d2UpSampling = tf.keras.layers.Conv3DTranspose(
        64, (1, 2, 2), (1, 2, 2), padding="same"
    )(d3)
    d2Concatenate = tf.keras.layers.Concatenate()([d2UpSampling, e2])
    d2 = Block(64, d2Concatenate)

    d1UpSampling = tf.keras.layers.Conv3DTranspose(
        32, (1, 2, 2), (1, 2, 2), padding="same"
    )(d2)
    d1Concatenate = tf.keras.layers.Concatenate()([d1UpSampling, e1])
    d1 = Block(32, d1Concatenate)

    d0UpSampling = tf.keras.layers.Conv3DTranspose(
        16, (1, 2, 2), (1, 2, 2), padding="same"
    )(d1)
    d0Concatenate = tf.keras.layers.Concatenate()([d0UpSampling, e0])
    d0 = Block(16, d0Concatenate)

    outputImage1 = SeperableConv(4, d0)

    # 보정 unet
    # 인코딩
    nextInputImage = Block(4, outputImage1)
    nextReferImage = Block(4, inputImage)
    concatenateImage = tf.keras.layers.Concatenate()([nextInputImage, nextReferImage])

    e0 = Block(16, concatenateImage)
    e0Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e0)

    e1 = Block(32, e0Pooling)
    e1Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e1)

    e2 = Block(64, e1Pooling)
    e2Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e2)

    e3 = Block(128, e2Pooling)
    e3Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e3)

    e4 = Block(256, e3Pooling)
    e4Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e4)

    # 중간
    bottleNeck = Block(512, e4Pooling)

    d4UpSampling = tf.keras.layers.Conv3DTranspose(
        256, (1, 2, 2), (1, 2, 2), padding="same"
    )(bottleNeck)
    d4Concatenate = tf.keras.layers.Concatenate()([d4UpSampling, e4])
    d4 = Block(256, d4Concatenate)

    d3UpSampling = tf.keras.layers.Conv3DTranspose(
        128, (1, 2, 2), (1, 2, 2), padding="same"
    )(d4)
    d3Concatenate = tf.keras.layers.Concatenate()([d3UpSampling, e3])
    d3 = Block(128, d3Concatenate)

    d2UpSampling = tf.keras.layers.Conv3DTranspose(
        64, (1, 2, 2), (1, 2, 2), padding="same"
    )(d3)
    d2Concatenate = tf.keras.layers.Concatenate()([d2UpSampling, e2])
    d2 = Block(64, d2Concatenate)

    d1UpSampling = tf.keras.layers.Conv3DTranspose(
        32, (1, 2, 2), (1, 2, 2), padding="same"
    )(d2)
    d1Concatenate = tf.keras.layers.Concatenate()([d1UpSampling, e1])
    d1 = Block(32, d1Concatenate)

    d0UpSampling = tf.keras.layers.Conv3DTranspose(
        16, (1, 2, 2), (1, 2, 2), padding="same"
    )(d1)
    d0Concatenate = tf.keras.layers.Concatenate()([d0UpSampling, e0])
    d0 = Block(16, d0Concatenate)

    outputImage2 = SeperableConv(4, d0)

    return tf.keras.Model(inputImage, [outputImage1, outputImage2])


if __name__ == "main":
    UnetModel().summary()
