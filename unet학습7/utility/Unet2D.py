import tensorflow as tf


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


def UnetModel(inputShape=(None, None, 4)):
    primaryImage = tf.keras.Input(shape=inputShape)
    secondaryImage = tf.keras.Input(shape=inputShape)

    ###

    ###
    # 프라이머리 인코더
    pe1 = Block(16, primaryImage)
    pe1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe1)

    pe2 = Block(32, pe1Pooling)
    pe2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe2)

    pe3 = Block(64, pe2Pooling)
    pe3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe3)

    pe4 = Block(128, pe3Pooling)
    pe4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(pe4)

    # 중간
    pbottleNeck = Block(256, pe4Pooling)

    # 디코더
    pd4UpSampling = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2, padding="same")(
        pbottleNeck
    )
    pd4Concatenate = tf.keras.layers.Concatenate()([pd4UpSampling, pe4])
    pd4 = Block(128, pd4Concatenate)

    pd3UpSampling = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2, padding="same")(pd4)
    pd3Concatenate = tf.keras.layers.Concatenate()([pd3UpSampling, pe3])
    pd3 = Block(64, pd3Concatenate)

    pd2UpSampling = tf.keras.layers.Conv2DTranspose(32, (2, 2), 2, padding="same")(pd3)
    pd2Concatenate = tf.keras.layers.Concatenate()([pd2UpSampling, pe2])
    pd2 = Block(32, pd2Concatenate)

    pd1UpSampling = tf.keras.layers.Conv2DTranspose(16, (2, 2), 2, padding="same")(pd2)
    pd1Concatenate = tf.keras.layers.Concatenate()([pd1UpSampling, pe1])
    pd1 = Block(16, pd1Concatenate)

    ###

    ###
    # 세컨더리 인코더
    se1 = Block(16, secondaryImage)
    se1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se1)

    se2 = Block(32, se1Pooling)
    se2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se2)

    se3 = Block(64, se2Pooling)
    se3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se3)

    se4 = Block(128, se3Pooling)
    se4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(se4)

    # 중간
    sbottleNeck = Block(256, se4Pooling)

    # 디코더
    sd4UpSampling = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2, padding="same")(
        sbottleNeck
    )
    sd4Concatenate = tf.keras.layers.Concatenate()([sd4UpSampling, se4])
    sd4 = Block(128, sd4Concatenate)

    sd3UpSampling = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2, padding="same")(sd4)
    sd3Concatenate = tf.keras.layers.Concatenate()([sd3UpSampling, se3])
    sd3 = Block(64, sd3Concatenate)

    sd2UpSampling = tf.keras.layers.Conv2DTranspose(32, (2, 2), 2, padding="same")(sd3)
    sd2Concatenate = tf.keras.layers.Concatenate()([sd2UpSampling, se2])
    sd2 = Block(32, sd2Concatenate)

    sd1UpSampling = tf.keras.layers.Conv2DTranspose(16, (2, 2), 2, padding="same")(sd2)
    sd1Concatenate = tf.keras.layers.Concatenate()([sd1UpSampling, se1])
    sd1 = Block(16, sd1Concatenate)

    ###

    ###
    # 백본 ae

    maine1 = Block(32, tf.keras.layers.Concatenate()([pe1, se1]))
    maine1Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(maine1)

    maine2 = Block(64, tf.keras.layers.Concatenate()([pe2, maine1Pooling, se2]))
    maine2Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(maine2)

    maine3 = Block(128, tf.keras.layers.Concatenate()([pe3, maine2Pooling, se3]))
    maine3Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(maine3)

    maine4 = Block(256, tf.keras.layers.Concatenate()([pe4, maine3Pooling, se4]))
    maine4Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(maine4)

    # 중간
    mainbottleNeck = Block(
        512, tf.keras.layers.Concatenate()([pbottleNeck, maine4Pooling, sbottleNeck])
    )

    # 디코더
    maind4UpSampling = tf.keras.layers.Conv2DTranspose(256, (2, 2), 2, padding="same")(
        mainbottleNeck
    )
    maind4Concatenate = tf.keras.layers.Concatenate()(
        [pd4, maind4UpSampling, maine4, sd4]
    )
    maind4 = Block(256, maind4Concatenate)

    maind3UpSampling = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2, padding="same")(
        maind4
    )
    maind3Concatenate = tf.keras.layers.Concatenate()(
        [pd3, maind3UpSampling, maine3, sd3]
    )
    maind3 = Block(128, maind3Concatenate)

    maind2UpSampling = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2, padding="same")(
        maind3
    )
    maind2Concatenate = tf.keras.layers.Concatenate()(
        [pd2, maind2UpSampling, maine2, sd2]
    )
    maind2 = Block(64, maind2Concatenate)

    maind1UpSampling = tf.keras.layers.Conv2DTranspose(32, (2, 2), 2, padding="same")(
        maind2
    )
    maind1Concatenate = tf.keras.layers.Concatenate()(
        [pd1, maind1UpSampling, maine1, sd1]
    )
    maind1 = Block(32, maind1Concatenate)

    outputImage = SeperableConv(4, maind1)

    return tf.keras.Model([primaryImage, secondaryImage], outputImage)


if __name__ == "main":
    laddernetModel().summary()
