import os
import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습4/generate/tester.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습4/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습5/checkPoint/20_checkPoint.ckpt"


x, y = DivideConcatenate(LoadGifExtract(imagePath))
# 모델생성
model = UnetModel(inputShape=(None, None, 12))

# model.load_weights(weightPath)
predImage = model.predict(x)[0]
predImage = np.clip(predImage, 0, 1) * 255
predImage = np.split(predImage, 2)


SaveGif(imageSavingPath, np.split(x * 255, 2, axis=-1), predImage)
