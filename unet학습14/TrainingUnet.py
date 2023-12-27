import tensorflow as tf
import os
from PIL import Image
import numpy as np
import random
from utility.ImageProcessing import *
from utility.Unet2D import *

pathTrain = "/content/drive/MyDrive/unet학습14/train"  # 학습할 이미지
pathTest = "/content/drive/MyDrive/unet학습14/test"  # 벨리데이션 테스트 이미지
pathSave = "/content/drive/MyDrive/unet학습14/checkPoint"  # 모델 저장할 위치
pathWeight = "/content/drive/MyDrive/unet학습14/checkPoint/10-cp.ckpt"

sample = 60
batchSize = 4
epoch = 10
lr = 1e-2
wd = 1e-4

rlrFactor = 0.5
rlrPatience = 8
rlrMinLr = 5e-5


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")

# tf.keras.mixed_precision.set_global_policy("mixed_float16")


print("")
print("트레인셋 전처리")
"""
# 파일 프레임 별로 전처리
for i in GetFilePath(pathTrain, end=".gif"):
    PreprocessGif(i, frame=6)
# 파일 사이즈별로 정리
DivideFolder(GetFilePath(pathTrain, end=".gif"), pathTrain, diviedSize=64)

"""
gifPath = GetFilePath(
    os.path.join(pathTrain, str(np.random.randint(1, 9) * 64)), end=".gif"
)
gifPath = random.sample(gifPath, sample)

trainDataset = tf.data.Dataset.from_generator(
    DatasetGenerater,
    args=[gifPath],
    output_types=(
        (
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        ),
        tf.float32,
    ),
    output_shapes=(
        (
            (None, None, 4),
            (None, None, 4),
            (None, None, 4),
            (None, None, 1),
        ),
        (None, None, 4),
    ),
)
# (inputImages, outputImages)
trainDataset = trainDataset.shuffle(4).batch(batchSize).prefetch(1)

print("")
print("총 트레인셋 갯수", len(gifPath))


print("")
print("테스트셋 전처리")
gifPathTest = GetFilePath(pathTest)
testDataset = tf.data.Dataset.from_generator(
    DatasetGenerater,
    args=[gifPathTest],
    output_types=(
        (
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        ),
        tf.float32,
    ),
    output_shapes=(
        (
            (None, None, 4),
            (None, None, 4),
            (None, None, 4),
            (None, None, 1),
        ),
        (None, None, 4),
    ),
)
# (inputImages, outputImages)
testDataset = testDataset.batch(batchSize).prefetch(1)

print("")
print("총 테스트셋 갯수", len(gifPathTest))


# 모델생성
model = vfiModel(inputShape=(None, None, 4))
model.load_weights(pathWeight)


# 콜백생성
cpCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(pathSave, "{epoch}-cp.ckpt"),
    verbose=1,
    save_weights_only=True,
)

# rlrCallback = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="loss", factor=rlrFactor, patience=rlrPatience, min_lr=rlrMinLr
# )

# 컴파일
model.compile(
    tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd),
    loss=tf.keras.losses.mean_absolute_error,
    metrics=["accuracy"],
)

model.fit(
    trainDataset,
    validation_batch_size=testDataset,
    epochs=epoch,
    callbacks=[cpCallback],
)
