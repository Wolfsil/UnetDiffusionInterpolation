import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습14/generate/testing.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습14/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습14/checkPoint/10-cp.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    if inputImage.shape[0] < 3:
        print("이미지 갯수가 부족합니다")
        return 0

    stepSize = 1.0 / diffusionStep

    noiseImage = np.random.rand(
        1,
        inputImage.shape[1],
        inputImage.shape[2],
        inputImage.shape[3],
    )

    ones = np.ones((1, inputImage.shape[1], inputImage.shape[2], 1))

    image1 = np.expand_dims(inputImage[0], axis=0)
    image2 = np.expand_dims(inputImage[1], axis=0)

    for step in range(diffusionStep):
        diffusionTime = 1.0 - step * stepSize
        sigRate, noiseRate = DiffusionSchedule(diffusionTime)
        step = ones * sigRate
        

        predNoise = model.predict([image1, image2, noiseImage, step])
        predImage = (noiseImage - noiseRate * predNoise) / sigRate

        sigRate, noiseRate = DiffusionSchedule(diffusionTime - stepSize)
        noiseImage = sigRate * predImage + noiseRate * predNoise

    return np.clip(predImage, 0, 1)


test = LoadGifExtract(imagePath)
# 모델생성
model = vfiModel(inputShape=(None, None, 4))
# 로드할 웨이트가 존재하면
model.load_weights(weightPath)

SaveGif(imageSavingPath, test, FrameInterpoloation(model, test))
