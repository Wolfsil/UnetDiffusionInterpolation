import tensorflow as tf


class Block(tf.keras.layers.Layer):
    def __init__(self, filter=512, kernelSize=3, name="Block"):
        super(Block, self).__init__(name=name)
        self.filter = filter
        self.kernelSize = kernelSize

    def build(self, input_shape):
        self.Conv1 = tf.keras.layers.SeparableConv2D(
            self.filter, kernel_size=self.kernelSize, padding="same"
        )
        self.Norm1 = tf.keras.layers.LayerNormalization()
        self.Activate1 = tf.keras.layers.Activation("swish")

        self.Conv2 = tf.keras.layers.SeparableConv2D(
            self.filter, kernel_size=self.kernelSize, padding="same"
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


class Unet(tf.keras.layers.Layer):
    def __init__(self, outputSize=4, name="Unet"):
        super(Unet, self).__init__(name=name)
        self.outputSize = outputSize

    def build(self, input_shape):
        self.BlockE1 = Block(32, 3, "E1")
        self.BlockE2 = Block(64, 3, "E2")
        self.BlockE3 = Block(128, 3, "E3")
        self.BlockE4 = Block(256, 3, "E4")
        self.BlockE5 = Block(512, 3, "E5")

        self.Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.Upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.Concatenate = tf.keras.layers.Concatenate()

        self.UpD4 = tf.keras.layers.SeparableConv2D(256, kernel_size=3, padding="same")
        self.BlockD4 = Block(256, 3, "D4")

        self.UpD3 = tf.keras.layers.SeparableConv2D(128, kernel_size=3, padding="same")
        self.BlockD3 = Block(128, 3, "D3")

        self.UpD2 = tf.keras.layers.SeparableConv2D(64, kernel_size=3, padding="same")
        self.BlockD2 = Block(64, 3, "D2")

        self.UpD1 = tf.keras.layers.SeparableConv2D(32, kernel_size=3, padding="same")
        self.BlockD1 = Block(32, 3, "D1")

        self.Output = Block(self.outputSize, 3, "UnetOutput")

    def call(self, input):
        e1 = self.BlockE1(input)
        e2 = self.Pooling(e1)

        e2 = self.BlockE2(e2)
        e3 = self.Pooling(e2)

        e3 = self.BlockE3(e3)
        e4 = self.Pooling(e3)

        e4 = self.BlockE4(e4)
        e5 = self.Pooling(e4)

        e5 = self.BlockE5(e5)

        d4 = self.Upsampling(e5)
        d4 = self.UpD4(d4)
        d4 = self.Concatenate([e4, d4])
        d4 = self.BlockD4(d4)

        d3 = self.Upsampling(d4)
        d3 = self.UpD3(d3)
        d3 = self.Concatenate([e3, d3])
        d3 = self.BlockD3(d3)

        d2 = self.Upsampling(d3)
        d2 = self.UpD2(d2)
        d2 = self.Concatenate([e2, d2])
        d2 = self.BlockD2(d2)

        d1 = self.Upsampling(d2)
        d1 = self.UpD1(d1)
        d1 = self.Concatenate([e1, d1])
        d1 = self.BlockD1(d1)

        output = self.Output(d1)

        return output


class Embedding(tf.keras.layers.Layer):
    def __init__(self, outputChannel=256, name="Embedding"):
        super(Embedding, self).__init__(name=name)
        self.outputChannel = outputChannel

    def build(self, input_shape):
        self.Resize512 = tf.keras.layers.Resizing(512, 512, "area")

        self.BlockE1 = Block(32, 3, "E1")
        self.BlockE2 = Block(64, 3, "E2")
        self.BlockE3 = Block(128, 3, "E3")
        self.BlockE4 = Block(256, 3, "E4")
        self.Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

        self.Flat = tf.keras.layers.Flatten()
        self.Dropout = tf.keras.layers.Dropout(0.1)
        self.DenseOutput = tf.keras.layers.Dense(self.outputChannel, activation="swish")
        self.Reshape = tf.keras.layers.Reshape((1, 1, -1))

    def call(self, inputs):
        e1 = self.Resize512(inputs)  # 512
        e1 = self.BlockE1(e1)
        e1 = self.Pooling(e1)

        e2 = self.BlockE2(e1)
        e2 = self.Pooling(e2)

        e3 = self.BlockE3(e2)
        e3 = self.Pooling(e3)

        e4 = self.BlockE4(e3)  # 64
        e4 = self.Pooling(e4)

        d = self.Flat(e4)
        d = self.Dropout(d)
        d = self.DenseOutput(d)
        d = self.Reshape(d)

        return d


def vfiModel(inputShape=(512, 512, 4)):
    # 입력값
    image1 = tf.keras.Input(shape=inputShape, name="1")
    image2 = tf.keras.Input(shape=inputShape, name="2")
    noisyImage = tf.keras.Input(shape=inputShape, name="noisyImage")
    step = tf.keras.Input(shape=(inputShape[0], inputShape[1], 1), name="step")

    # 임베딩
    embedding1 = Embedding(outputChannel=256)(
        tf.keras.layers.Concatenate()([image1, image2])
    )  # (None, 1, 1, 256)
    embedding2 = Block(256, 1)(
        tf.keras.layers.Concatenate()([image1, image2, noisyImage])
    )  # (None, 가로, 세로, 256)
    embedding = embedding1 + embedding2  # (None, 가로, 세로, 512)

    # unet
    predict = Unet(4, name="preprocessing")(embedding)

    return tf.keras.Model([image1, image2, noisyImage, step], predict)


if __name__ == "main":
    vfiModel().summary()
