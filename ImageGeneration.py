import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet3D import *

imagePath = "/content/drive/MyDrive/unet학습/generate"
imageSavingPath = "/content/drive/MyDrive/unet학습/generate"
weightPath = "/content/drive/MyDrive/unet학습/checkPoint/cp-20.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    if inputImage.shape[0] < 2:
        print("이미지 갯수가 부족합니다")
        return 0

    stepSize = 1.0 / diffusionStep

    noiseImage = np.random.rand(
        inputImage.shape[0],
        inputImage.shape[1],
        inputImage.shape[2],
        inputImage.shape[3],
    )
    ones = np.ones((inputImage.shape[0], inputImage.shape[1], inputImage.shape[2], 1))

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

    return np.clip(predImage, 0, 1)


test = LoadGifAll(imagePath)
# 모델생성
model = UnetModel(inputShape=(None, None, None, 9))
# 로드할 웨이트가 존재하면
model.load_weights(weightPath)

SaveGif(imageSavingPath, FrameInterpoloation(model, test))
