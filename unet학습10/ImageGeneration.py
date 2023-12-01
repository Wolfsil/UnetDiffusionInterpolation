import numpy as np

from utility.ImageProcessing import *
from utility.Unet2D import *

imagePath = "/content/drive/MyDrive/unet학습10/test/test2.gif"
imageSavingPath = "/content/drive/MyDrive/unet학습10/generate/return.gif"
weightPath = "/content/drive/MyDrive/unet학습10/checkPoint/20_checkPoint.ckpt"


# 프레임보간 함수
def FrameInterpoloation(model, inputImage: np.ndarray):
    predImage = model.predict([inputImage[0:1], inputImage[1:2]])[0]

    return np.clip(predImage, 0, 255)


x, y = Divide(LoadGifExtract(imagePath))
# 모델생성
model = UnetModel(inputShape=(None, None, 4))

model.load_weights(weightPath)

SaveGif(imageSavingPath, x, FrameInterpoloation(model, x))
