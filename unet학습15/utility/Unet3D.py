import tensorflow as tf


# def SeperableConv(filter, input):
#     depthwise = tf.keras.layers.Conv3D(
#         input.shape[-1], 3, padding="same", groups=input.shape[-1]
#     )(input)
#     pointwise = tf.keras.layers.Conv3D(filter, 1, padding="same")(depthwise)
#     return pointwise


# def SeperableConvEnd(filter, input):
#     depthwise = tf.keras.layers.Conv3D(
#         input.shape[-1], 3, padding="same", groups=input.shape[-1]
#     )(input)
#     pointwise = tf.keras.layers.Conv3D(filter, 1, padding="same", dtype=tf.float32)(
#         depthwise
#     )
#     return pointwise


class SeperableConv3D(tf.keras.layers.Layer):
    def __init__(self, filter=512, kernelSize=3, padding="same", name="Block"):
        super(SeperableConv3D, self).__init__(name=name)
        self.filter = filter
        self.kernelSize = kernelSize
        self.padding = padding

    def build(self, input_shape):
        self.depthwise = tf.keras.layers.Conv3D(
            input_shape[-1],
            self.kernelSize,
            padding=self.padding,
            groups=input_shape[-1],
        )
        self.pointwise = tf.keras.layers.Conv3D(self.filter, 1, padding=self.padding)

    def call(self, input):
        input = self.depthwise(input)
        input = self.pointwise(input)
        return input


class Block(tf.keras.layers.Layer):
    def __init__(self, filter=512, kernelSize=3, name="Block"):
        super(Block, self).__init__(name=name)
        self.filter = filter
        self.kernelSize = kernelSize

    def build(self, input_shape):
        self.Conv1 = SeperableConv3D(
            self.filter, kernelSize=self.kernelSize, padding="same"
        )
        self.Norm1 = tf.keras.layers.LayerNormalization()
        self.Activate1 = tf.keras.layers.Activation("swish")

        self.Conv2 = SeperableConv3D(
            self.filter, kernelSize=self.kernelSize, padding="same"
        )
        self.Norm2 = tf.keras.layers.LayerNormalization()
        self.Activate2 = tf.keras.layers.Activation("swish")

    def call(self, input):
        input = self.Conv1(input)
        input = self.Norm1(input)
        input = self.Activate1(input)
        input = self.Conv2(input)
        input = self.Norm2(input)
        input = self.Activate2(input)

        return input


def UnetModel(inputShape=(None, None, None, 4)):
    inputImage = tf.keras.Input(shape=inputShape)
    step = tf.keras.Input(shape=(inputShape[0], inputShape[1], inputShape[2], 1))

    # 인코딩
    e1 = Block(32, name="e1")(tf.keras.layers.Concatenate()([inputImage, step]))
    e1Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e1)

    e2 = Block(64, name="e2")(e1Pooling)
    e2Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e2)

    e3 = Block(128, name="e3")(e2Pooling)
    e3Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e3)

    e4 = Block(256, name="e4")(e3Pooling)
    e4Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(e4)

    # 중간
    bottleNeck = Block(512, name="bottle")(e4Pooling)

    d4UpSampling = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(bottleNeck)
    d4Transpose = SeperableConv3D(256, name="d4_T")(d4UpSampling)
    d4Concatenate = tf.keras.layers.Concatenate()([d4Transpose, e4])
    d4 = Block(256, name="d4")(d4Concatenate)

    d3UpSampling = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(d4)
    d3Transpose = SeperableConv3D(128, name="d3_T")(d3UpSampling)
    d3Concatenate = tf.keras.layers.Concatenate()([d3Transpose, e3])
    d3 = Block(128, name="d3")(d3Concatenate)

    d2UpSampling = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(d3)
    d2Transpose = SeperableConv3D(64, name="d2_T")(d2UpSampling)
    d2Concatenate = tf.keras.layers.Concatenate()([d2Transpose, e2])
    d2 = Block(64, name="d2")(d2Concatenate)

    d1UpSampling = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(d2)
    d1Transpose = SeperableConv3D(32, name="d1_T")(d1UpSampling)
    d1Concatenate = tf.keras.layers.Concatenate()([d1Transpose, e1])
    d1 = Block(32, name="d1")(d1Concatenate)

    outputImage = SeperableConv3D(4, name="output")(d1)

    return tf.keras.Model([inputImage, step], outputImage)


if __name__ == "main":
    UnetModel().summary()
