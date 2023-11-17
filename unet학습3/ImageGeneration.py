import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습/generate/test.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습3/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습3/checkPoint/20_checkPoint.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    stepSize = 1.0 / diffusionStep

    noiseImage = np.random.rand(
        inputImage.shape[0],
        inputImage.shape[1],
        8,
    )

    ones = np.ones((inputImage.shape[0], inputImage.shape[1], 1))

    for step in range(diffusionStep):
        diffusionTime = 1.0 - step * stepSize
        sigRate, noiseRate = DiffusionSchedule(diffusionTime)
        step = ones * sigRate
        concatenateImage = np.concatenate([inputImage, noiseImage, step], axis=-1)
        concatenateImage = np.expand_dims(concatenateImage, axis=0)

        predNoise = model.predict(concatenateImage)[0]
        predImage = (noiseImage - noiseRate * predNoise) / sigRate

        sigRate, noiseRate = DiffusionSchedule(diffusionTime - stepSize)
        noiseImage = sigRate * predImage + noiseRate * predNoise

    return np.split(np.clip(predImage, 0, 1), 2, axis=-1)


test = LoadGifExtract(imagePath)
# 모델생성
model = UnetModel(inputShape=(None, None, 21))

# model.load_weights(weightPath)

SaveGif(imageSavingPath, test, FrameInterpoloation(model, test))
