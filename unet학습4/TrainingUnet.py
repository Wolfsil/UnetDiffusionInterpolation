import tensorflow as tf
import os
from PIL import Image
import numpy as np
import random
from utility.ImageProcessing import *
from utility.Unet2D import *

pathTrain = "/content/drive/MyDrive/unet학습/train"  # 학습할 이미지
pathTest = "/content/drive/MyDrive/unet학습/test"  # 벨리데이션 테스트 이미지
pathSave = (
    "/content/drive/MyDrive/unet학습3/checkPoint/{epoch}_checkPoint.ckpt"  # 모델 저장할 위치
)
pathWeight = "/content/drive/MyDrive/unet학습3/checkPoint/20_checkPoint.ckpt"

sample = 112
batchSize = 1
epoch = 20
lr = 1e-3
wd = 1e-4

rlrFactor = 0.5
rlrPatience = 7
rlrMinLr = 1e-4


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")

# tf.keras.mixed_precision.set_global_policy("mixed_float16")


print("")
print("트레인셋 전처리")
# 사용불가능 파일 전처리
gifPath = GetFilePath(pathTrain)
for i in gifPath:
    PreprocessGif(i)
gifPath = GetFilePath(pathTrain)
gifPath = random.sample(gifPath, sample)

print("")
print("총 트레인셋 갯수", len(gifPath))

trainDataset = tf.data.Dataset.from_generator(
    DatasetGenerater,
    args=[gifPath],
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, None, 13), (None, None, 4)),
)
# (inputImages, outputImages)
trainDataset = trainDataset.batch(batchSize).prefetch(1)

# print("")
# print("테스트셋 전처리")
# gifPathTest = GetFilePath(pathTest)
# for i in gifPathTest:
#     PreprocessGif(i)
# gifPathTest = GetFilePath(pathTest)

# print("")
# print("총 테스트셋 갯수", len(gifPathTest))


# testDataset = tf.data.Dataset.from_generator(
#     DatasetGenerater,
#     args=[gifPathTest],
#     output_types=(tf.float32, tf.float32),
#     output_shapes=((None, None, 13), (None, None, 4)),
# )
# # (inputImages, outputImages)
# testDataset = testDataset.batch(batchSize).prefetch(1)


# 모델생성
model = UnetModel(inputShape=(None, None, 13))
# model.load_weights(pathWeight)

# 콜백생성
cpCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=pathSave,
    verbose=1,
    save_weights_only=True,
)
rlrCallback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=rlrFactor, patience=rlrPatience, min_lr=rlrMinLr
)

# 컴파일
model.compile(
    tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd),
    loss=tf.keras.losses.mean_absolute_error,
    metrics=["accuracy"],
)

model.fit(
    trainDataset,
    epochs=epoch,
    callbacks=[
        cpCallback
        #  , rlrCallback
    ],
)
