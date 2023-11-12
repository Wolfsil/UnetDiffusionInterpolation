import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet3D import *


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
        sigRate, noiseRate = diffusionSchedule(diffusionTime)
        step = ones * sigRate
        concatenateImage = np.concatenate([inputImage, noiseImage, step], axis=-1)
        concatenateImage = np.expand_dims(concatenateImage, axis=0)

        predNoise = model.predict(concatenateImage)[0]
        predImage = (noiseImage - noiseRate * predNoise) / sigRate

        sigRate, noiseRate = diffusionSchedule(diffusionTime - stepSize)
        noiseImage = sigRate * predImage + noiseRate * predNoise

    return np.clip(predImage, 0, 1)


test = LoadGifAll("C:/Users/82109/Desktop/unet학습/testerDivided.gif")
# 모델생성
model = UnetModel(inputShape=(None, None, None, 9))
# 로드할 웨이트가 존재하면
model.load_weights(
    os.path.join("C:\\Users\\82109\\Desktop\\unet학습\\checkPoint\\8cp-1.ckpt")
)

saveGif(
    "C:/Users/82109/Desktop/unet학습/testerGenerate.gif", FrameInterpoloation(model, test)
)
