from ImageProcessing import *


a = LoadGifAll("C:/Users/82109/Desktop/unet학습/testerDivided.gif")
a = a[0::2]
SaveGif("C:/Users/82109/Desktop/unet학습/testerDivided.gif", a)
