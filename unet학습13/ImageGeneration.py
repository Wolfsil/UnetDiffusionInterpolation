import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습13/generate/testing.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습13/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습13/checkPoint/20-cp.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    if inputImage.shape[0] < 3:
        print("이미지 갯수가 부족합니다")
        return 0

    stepSize = 1.0 / diffusionStep

    noiseImage1 = np.random.rand(
        1,
        inputImage.shape[1],
        inputImage.shape[2],
        inputImage.shape[3],
    )
    noiseImage2 = np.random.rand(
        1,
        inputImage.shape[1],
        inputImage.shape[2],
        inputImage.shape[3],
    )

    ones = np.ones((1, inputImage.shape[1], inputImage.shape[2], 1))

    for step in range(diffusionStep):
        diffusionTime = 1.0 - step * stepSize
        sigRate, noiseRate = DiffusionSchedule(diffusionTime)
        step = ones * sigRate

        image1 = np.expand_dims(inputImage[0], axis=0)
        image2 = np.expand_dims(inputImage[1], axis=0)
        image3 = np.expand_dims(inputImage[2], axis=0)

        predNoise = model.predict(
            [image1, image2, image3, noiseImage1, noiseImage2, step, ones]
        )[0]
        predImage1 = (noiseImage1 - noiseRate * predNoise[:,:,0:4]) / sigRate
        predImage2 = (noiseImage2 - noiseRate * predNoise[:,:,4:8]) / sigRate

        sigRate, noiseRate = DiffusionSchedule(diffusionTime - stepSize)
        noiseImage1 = sigRate * predImage1 + noiseRate * predNoise[:,:,0:4]
        noiseImage2 = sigRate * predImage2 + noiseRate *  predNoise[:,:,4:8]

    return (np.clip(predImage1, 0, 1), np.clip(predImage2, 0, 1))


test = LoadGifExtract(imagePath)
# 모델생성
model = vfiModel(inputShape=(None, None, 4))
# 로드할 웨이트가 존재하면
model.load_weights(weightPath)

SaveGif(imageSavingPath, test, FrameInterpoloation(model, test))
