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
        self.BlockE1 = Block(32, 9, "E1")
        self.BlockE2 = Block(64, 7, "E2")
        self.BlockE3 = Block(128, 5, "E3")
        self.BlockE4 = Block(256, 3, "E4")
        self.BlockE5 = Block(512, 3, "E5")

        self.Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.Upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.Concatenate = tf.keras.layers.Concatenate()

        self.UpD4 = tf.keras.layers.SeparableConv2D(256, kernel_size=3, padding="same")
        self.BlockD4 = Block(256, 3, "D4")

        self.UpD3 = tf.keras.layers.SeparableConv2D(128, kernel_size=5, padding="same")
        self.BlockD3 = Block(128, 5, "D3")

        self.UpD2 = tf.keras.layers.SeparableConv2D(64, kernel_size=7, padding="same")
        self.BlockD2 = Block(64, 7, "D2")

        self.UpD1 = tf.keras.layers.SeparableConv2D(32, kernel_size=9, padding="same")
        self.BlockD1 = Block(32, 9, "D1")

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


class Divider(tf.keras.layers.Layer):
    def __init__(self, size=4, name="Divider"):
        super(Divider, self).__init__(name=name)
        self.size = size

    def call(self, input):
        return [input[:, :, :, 0 : self.size], input[:, :, :, self.size :]]


class Embedding(tf.keras.layers.Layer):
    def __init__(self, outputChannel=256, name="Embedding"):
        super(Embedding, self).__init__(name=name)
        self.outputChannel = outputChannel

    def build(self, input_shape):
        self.Resize256 = tf.keras.layers.Resizing(256, 256, "area")
        self.Resize128 = tf.keras.layers.Resizing(128, 128, "area")
        self.Resize64 = tf.keras.layers.Resizing(64, 64, "area")

        self.BlockE1 = Block(16, 9, "E1")
        self.BlockE2 = Block(32, 7, "E2")
        self.BlockE3 = Block(64, 5, "E3")
        self.BlockE4 = Block(128, 3, "E4")
        self.BlockE5 = Block(self.outputChannel, 3, "E5")
        self.BlockE6 = Block(self.outputChannel, 3, "E6")
        self.BlockE7 = Block(self.outputChannel, 3, "E7")

        self.Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.Concatenate = tf.keras.layers.Concatenate()

        self.Output = Block(self.outputChannel, 3, "EmbeddingOutput")

    def call(self, inputs):
        concateImages = self.Concatenate(
            [
                inputs[0],
                inputs[1],
                inputs[2],
            ]
        )
        e1 = self.Resize256(concateImages)  # 256
        e1 = self.BlockE1(e1)
        e1 = self.Pooling(e1)

        e2 = self.Resize128(concateImages)  # 128
        e2 = self.Concatenate([e1, e2])
        e2 = self.BlockE2(e2)
        e2 = self.Pooling(e2)

        e3 = self.Resize64(concateImages)  # 64
        e3 = self.Concatenate([e2, e3])
        e3 = self.BlockE3(e3)
        e3 = self.Pooling(e3)

        e4 = self.BlockE4(e3)  # 32
        e4 = self.Pooling(e4)

        e5 = self.BlockE5(e4)  # 16
        e5 = self.Pooling(e5)

        e6 = self.BlockE6(e5)  # 8
        e6 = self.Pooling(e6)

        e7 = self.BlockE7(e6)  # 4
        e7 = self.Pooling(e7)

        output = self.Output(e7)  # 2
        output = self.Pooling(output)  # [배치, 세로(1), 가로(1), 채널]
        output = inputs[3] * output
        return output


def vfiModel(inputShape=(512, 512, 4)):
    image1 = tf.keras.Input(shape=inputShape, name="1")
    image2 = tf.keras.Input(shape=inputShape, name="2")
    image3 = tf.keras.Input(shape=inputShape, name="3")
    noisyImage1 = tf.keras.Input(shape=inputShape, name="n1")
    noisyImage2 = tf.keras.Input(shape=inputShape, name="n2")
    step = tf.keras.Input(shape=(inputShape[0], inputShape[1], 1), name="step")
    ones= tf.keras.Input(shape=(inputShape[0], inputShape[1], 1), name="ones")

    embedding = Embedding(outputChannel=256)([image1, image2, image3, ones])

    imageConcatenate = tf.keras.layers.Concatenate()(
        [image1, image2, image3, noisyImage1, noisyImage2, step, embedding]
    )
    predict = Unet(8, name="preprocessing")(imageConcatenate)

    return tf.keras.Model(
        [image1, image2, image3, noisyImage1, noisyImage2, step,ones], predict 
    )


if __name__ == "main":
    vfiModel().summary()
