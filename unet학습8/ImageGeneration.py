import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습8/generate/tester.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습8/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습8/checkPoint/20_checkPoint.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    stepSize = 1.0 / diffusionStep

    noiseImage = np.random.rand(
        1,
        inputImage.shape[1],
        inputImage.shape[2],
        4,
    )*255
    ones = np.ones((1, inputImage.shape[1], inputImage.shape[2], 1))

    # predImage = model.predict([inputImage[0:1], inputImage[1:2]])[0]

    for step in range(diffusionStep):
        diffusionTime = 1.0 - step * stepSize
        sigRate, noiseRate = DiffusionSchedule(diffusionTime)
        step = ones * sigRate

        predNoise = model.predict([inputImage[0:1], noiseImage, step, inputImage[1:2]])

        predImage = (noiseImage - noiseRate * predNoise) / sigRate

        sigRate, noiseRate = DiffusionSchedule(diffusionTime - stepSize)
        noiseImage = sigRate * predImage + noiseRate * predNoise
    return np.clip(predImage, 0, 255)


x, y = Divide(LoadGifExtract(imagePath))
# 모델생성
model = UnetModel(inputShape=(None, None, 4))

model.load_weights(weightPath)

SaveGif(imageSavingPath, x, FrameInterpoloation(model, x))
