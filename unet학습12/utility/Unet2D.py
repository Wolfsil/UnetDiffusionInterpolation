import tensorflow as tf
import tensorflow_addons as tfa


class Warping(tf.keras.layers.Layer):
    def __init__(self):
        super(Warping, self).__init__()
        self.Warp = tfa.image.dense_image_warp

    def call(self, inputs):  # 모델의 input과 output 계산방식 선언
        return self.Warp(inputs[0], inputs[1])


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
        self.BlockE1 = Block(16, 9, "E1")
        self.BlockE2 = Block(32, 7, "E2")
        self.BlockE3 = Block(64, 5, "E3")
        self.BlockE4 = Block(128, 3, "E4")
        self.BlockE5 = Block(256, 3, "E5")

        self.Pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.Upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.Concatenate = tf.keras.layers.Concatenate()

        self.UpD4 = tf.keras.layers.SeparableConv2D(128, kernel_size=3, padding="same")
        self.BlockD4 = Block(128, 3, "D4")

        self.UpD3 = tf.keras.layers.SeparableConv2D(64, kernel_size=5, padding="same")
        self.BlockD3 = Block(64, 5, "D3")

        self.UpD2 = tf.keras.layers.SeparableConv2D(32, kernel_size=7, padding="same")
        self.BlockD2 = Block(32, 7, "D2")

        self.UpD1 = tf.keras.layers.SeparableConv2D(16, kernel_size=9, padding="same")
        self.BlockD1 = Block(16, 9, "D1")

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


class OpticalFlowWarp(tf.keras.layers.Layer):
    def __init__(self, name="OpticalFlowWarp"):
        super(OpticalFlowWarp, self).__init__(name=name)

    def build(self, input_shape):
        self.Warp = Warping()
        self.Interp = Unet(5, "InterpUnet")
        self.Concatenate = tf.keras.layers.Concatenate()

    def call(self, inputs):
        image1, image2, flows = inputs
        flow1 = flows[:, :, :, 0:2]
        flow2 = flows[:, :, :, 2:4]

        warpImage1 = self.Warp([image1, flow1])
        warpImage2 = self.Warp([image2, flow2])

        flowInterp = self.Interp(
            self.Concatenate([image1, image2, flow1, flow2, warpImage1, warpImage2])
        )
        deltaFlow1 = flowInterp[:, :, :, 0:2] + flow1
        deltaFlow2 = flowInterp[:, :, :, 2:4] + flow2
        visibilityMap1 = tf.keras.layers.Activation("sigmoid")(flowInterp[:, :, :, 4:5])
        visibilityMap2 = 1 - visibilityMap1

        warpImage1 = self.Warp([image1, deltaFlow1])
        warpImage2 = self.Warp([image2, deltaFlow2])

        output = warpImage1 * visibilityMap1 + warpImage2 * visibilityMap2

        return output


def vfiModel(inputShape=(128, 128, 4)):
    image1 = tf.keras.Input(shape=inputShape)
    image2 = tf.keras.Input(shape=inputShape)

    imageConcatenate = tf.keras.layers.Concatenate()([image1, image2])
    preprocessing = Unet(8, name="preprocessing")(imageConcatenate)

    flows = Unet(4, name="flow")(preprocessing)
    predict = OpticalFlowWarp()([image1, image2, flows])

    return tf.keras.Model([image1, image2], predict)


if __name__ == "main":
    vfiModel().summary()
