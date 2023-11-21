import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet3D import *

imagePath = "/content/drive/MyDrive/unet학습/generate"
imageSavingPath1 = "/content/drive/MyDrive/unet학습/generate/return1.gif"
imageSavingPath2 = "/content/drive/MyDrive/unet학습/generate/return2.gif"
weightPath = "/content/drive/MyDrive/unet학습/checkPoint/cp-20.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray, diffusionStep=20):
    predImage = model.predict(inputImage)
    predImage1 = np.clip(predImage[0][0], 0, 255)
    predImage2 = np.clip(predImage[0][1], 0, 255)
    return predImage1, predImage2


test = LoadGifAll(imagePath)
# 모델생성
model = UnetModel(inputShape=(None, None, None, 4))
# 로드할 웨이트가 존재하면
model.load_weights(weightPath)
predImage = FrameInterpoloation(model, test)
SaveGif(imageSavingPath1, test, predImage[0])
SaveGif(imageSavingPath2, test, predImage[1])
