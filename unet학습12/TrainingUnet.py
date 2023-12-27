import tensorflow as tf
import os
from PIL import Image
import numpy as np
import random
from utility.ImageProcessing import *
from utility.Unet2D import *

pathTrain = "/content/drive/MyDrive/unet학습12/train/" + str(
    np.random.randint(1, 9) * 64
)  # 학습할 이미지
pathTest = "/content/drive/MyDrive/unet학습12/test"  # 벨리데이션 테스트 이미지
pathSave = "/content/drive/MyDrive/unet학습12/checkPoint/{epoch}_checkPoint.ckpt"  # 모델 저장할 위치
pathWeight = "/content/drive/MyDrive/unet학습12/checkPoint/20_checkPoint.ckpt"

sample = 68
batchSize = 4
epoch = 20
lr = 1e-3
wd = 1e-4

rlrFactor = 10
rlrPatience = 8
rlrMinLr = 1e-5
str(np.random.randint(1, 9) * 64)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")


print("")
print("트레인셋 전처리")
gifPath = GetFilePath(pathTrain)
gifPath = random.sample(gifPath, sample)
print("")
print("총 트레인셋 갯수", len(gifPath))

trainDataset = tf.data.Dataset.from_generator(
    DatasetGenerater,
    args=[gifPath],
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        ((None, None, 4), (None, None, 4)),
        (None, None, 4),
    ),
)
# (inputImages, outputImages)
trainDataset = trainDataset.batch(batchSize).prefetch(1)

print("")
print("테스트셋 전처리")
gifPathTest = GetFilePath(pathTest)
print("")
print("총 테스트셋 갯수", len(gifPathTest))


testDataset = tf.data.Dataset.from_generator(
    DatasetGenerater,
    args=[gifPathTest],
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        ((None, None, 4), (None, None, 4)),
        (None, None, 4),
    ),
)
# (inputImages, outputImages)
testDataset = testDataset.batch(1).prefetch(1)


# 모델생성
model = vfiModel(inputShape=(None, None, 4))
model.load_weights(pathWeight)

# 콜백생성
cpCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=pathSave,
    verbose=1,
    save_weights_only=True,
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
    validation_data=testDataset,
    callbacks=[cpCallback],
)
