import os
from PIL import Image
import numpy as np
import shutil


# 경로 반환(path: 파일들이 들어있는 폴더경로, end: 확장자)
def GetFilePath(path, end=".gif"):
    gifFileList = os.listdir(path)
    gifPath = []
    for name in gifFileList:
        if name.endswith(end):
            gifPath.append(os.path.join(path, name))
    return gifPath


# 못쓰는 데이터를 걸러줌(path: 파일의 경로)
def PreprocessGif(path, frame=6):
    gif = Image.open(path)
    targetFrame = gif.n_frames
    targetSize = gif.size  # 이미지 크기가 너무크면 오버플로우 발생함
    gif.close()
    if targetSize[0] > targetSize[1]:
        targetSize = targetSize[0]
    else:
        targetSize = targetSize[1]

    if targetFrame < frame or targetSize > 512:
        print(path, ": ", targetFrame, " ", targetSize, "사용불가능")
        os.remove(path=path)
    else:
        print(path, ": ", targetFrame, " ", targetSize, " 사용가능")


# 데이터를 사이즈 폴더별로 정리(pathList: 파일의 경로 배열, savePath: 저장할 폴더 경로, dividedSize: 얼마의 배수로 나눌것인지)
def DivideFolder(pathList, savePath, diviedSize=64):
    tag = 1
    for room in range(64, 513, 64):
        if not os.path.isdir(os.path.join(savePath, str(room))):
            os.mkdir(os.path.join(savePath, str(room)))

    for name in pathList:
        print(name)
        gif = Image.open(name)
        targetSize = gif.size
        gif.close()
        if targetSize[0] > targetSize[1]:
            targetSize = targetSize[0]
        else:
            targetSize = targetSize[1]

        room = ((targetSize - 1) // diviedSize) * diviedSize + diviedSize

        shutil.move(name, os.path.join(savePath, str(room), str(tag) + ".gif"))
        tag = tag + 1


# gif를 읽고 넘파이 배열로 노멀라이즈해줌(path: 이미지 파일 경로, extractFrame: 몇프레임을 뽑을 것인지, paddingSize: 몇 배수로 패딩)
def LoadGifExtract(path, extractFrame=3, paddingSize=64):
    gif = Image.open(path)
    flip = np.random.randint(0, 7)
    # extractFrame = np.random.randint(4, 9, 2)
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
        if temp.shape[0] > temp.shape[1]:
            temp = np.pad(
                temp,
                pad_width=((0, 0), (temp.shape[0] - temp.shape[1], 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif temp.shape[0] < temp.shape[1]:
            temp = np.pad(
                temp,
                pad_width=((temp.shape[1] - temp.shape[0], 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        images.append(temp)
    gif.close()
    return np.array(images) / 255.0


# (path: 이미지 파일 경로, paddingSize: 몇배수로 패딩할건지)
def LoadGifAll(path, paddingSize=64):
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
        if temp.shape[0] > temp.shape[1]:
            temp = np.pad(
                temp,
                pad_width=((0, 0), (0, temp.shape[0] - temp.shape[1]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif temp.shape[0] < temp.shape[1]:
            temp = np.pad(
                temp,
                pad_width=((0, temp.shape[1] - temp.shape[0]), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        images.append(temp)
    gif.close()
    return np.array(images) / 255.0


# 0번 차원에서 인풋데이터와 아웃풋 데이터를 분리(arr: 배열)
# def Divide(arr):
#     evens = arr[0::2]
#     odds = arr[1::2]
#     return evens, odds


# sincos 스케줄러(diffusionTime: 0~1까지의 수. 1에가까울수록 노이즈가 강함)
def DiffusionSchedule(diffusionTime):
    noiseAng = np.arccos(0.01)  # 90도
    sigAng = np.arccos(0.99)  # 0도
    diffusionAng = sigAng + diffusionTime * (
        noiseAng - sigAng
    )  # DFT가 1에 가까울수록 노이즈(1에서 시작)
    sigRate = np.cos(diffusionAng)  # DFT가 1에 가까울수록 0.01
    noiseRate = np.sin(diffusionAng)  # DFT가 1에 가까울수록 0.99
    return sigRate, noiseRate


# 데이터셋 제너레이터 생성
def DatasetGenerater(gifPath):
    # gif파일을 반환
    for i in gifPath:
        x = LoadGifExtract(i, 5)
        noise = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        step = np.ones((x.shape[0], x.shape[1], x.shape[2], 1))
        noise[1] = 0
        noise[3] = 0

        sigRate, noiseRate = DiffusionSchedule(np.random.rand())

        noisyImage = sigRate * x + noiseRate * noise

        noisyImage[1] = x[1]
        noisyImage[3] = x[3]

        step = step * sigRate

        yield (noisyImage, step), noise


def SaveGif(path, images):
    imgs = []
    for i in images:
        img = Image.fromarray((i * 255).round().astype(np.int8), mode="RGBA")
        imgs.append(img)
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:], disposal=2, duration=150, loop=0
    )
