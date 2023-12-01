import os
from PIL import Image
import numpy as np


# 경로 반환
def GetFilePath(path, end=".gif"):
    gifFileList = os.listdir(path)
    gifPath = []
    for name in gifFileList:
        if name.endswith(tuple(end)):
            gifPath.append(os.path.join(path, name))
    return gifPath


# 못쓰는 데이터를 걸러줌
def PreprocessGif(path, frame=4):
    gif = Image.open(path)
    targetFrame = gif.n_frames
    targetSize = gif.size  # 이미지 크기가 너무크면 오버플로우 발생함
    gif.close()
    if targetSize[0] > targetSize[1]:
        targetSize = targetSize[0]
    else:
        targetSize = targetSize[1]

    if targetFrame < frame or targetSize > 448 or targetSize < 256 :
        print(path, ": ", targetFrame, " ", targetSize, "사용불가능")
        os.remove(path=path)
    else:
        print(path, ": ", targetFrame, " ", targetSize, " 사용가능")


# gif를 읽고 넘파이 배열로 노멀라이즈해줌
def LoadGifExtract(path, extractFrame=3, paddingSize=32):
    gif = Image.open(path)
    flip = np.random.randint(0, 7)
    remainFrame = gif.n_frames - extractFrame
    start = 0
    end = 0
    if remainFrame <= 1:
        start = 1
        end = gif.n_frames
    else:
        start = np.random.randint(1, remainFrame + 1)
        end = start + extractFrame
    images = []
    for i in range(start, end):
        gif.seek(i)
        temp = gif.transpose(flip).convert("RGBA")
        temp = np.array(temp)
        height = (paddingSize - temp.shape[0] % paddingSize) % paddingSize
        width = (paddingSize - temp.shape[1] % paddingSize) % paddingSize
        temp = np.pad(
            temp,
            pad_width=((0, height), (0, width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        images.append(temp)
    gif.close()
    return np.array(images)


def LoadGifAll(path, paddingSize=32):
    gif = Image.open(path)

    images = []
    for i in range(1, gif.n_frames):
        gif.seek(i)
        temp = gif.convert("RGBA")
        temp = np.array(temp)
        height = (paddingSize - temp.shape[0] % paddingSize) % paddingSize
        width = (paddingSize - temp.shape[1] % paddingSize) % paddingSize
        temp = np.pad(
            temp,
            pad_width=((0, height), (0, width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        images.append(temp)
    gif.close()
    return np.array(images)


# # 인풋데이터와 아웃풋 데이터를 분리
def Divide(arr):
    evens = arr[0::2]
    odds = arr[1::2]
    return evens, odds


# 데이터셋 제너레이터 생성
def DatasetGenerater(gifPath):
    # gif파일을 반환
    for i in gifPath:
        inputImage, outputImage = Divide(LoadGifExtract(i))  # 4채널(2개), 4채널(1개)

        yield (inputImage[0], inputImage[1]), outputImage[0]


# 아름답진 않지만 간결하다.
def SaveGif(path, inputImage, outputImage):
    imgs = []

    inputImage = inputImage
    outputImage = outputImage
    imgs.append(Image.fromarray(inputImage[0].round().astype(np.int8), mode="RGBA"))
    imgs.append(Image.fromarray(outputImage.round().astype(np.int8), mode="RGBA"))

    imgs.append(Image.fromarray(inputImage[1].round().astype(np.int8), mode="RGBA"))

    imgs[0].save(
        path, save_all=True, append_images=imgs[1:], disposal=2, duration=400, loop=0
    )
