import os
from PIL import Image
import numpy as np


path=""
def loadGif(path):
  gif=Image.open(path)
  print(gif.n_frames)
  print(gif.size)
  images=[]
  for i in range(gif.n_frames):
    gif.seek(i)
    image=np.array(gif.convert("RGBA"))
    images.append(image)
  return np.array(images)


#못쓰는 데이터를 걸러줌
def preprocessGif(path,frame=4):
  gif=Image.open(path)
  if gif.n_frames<frame:
    print(path,": ",gif.n_frames,"사용불가능")
    os.remove(path=path)
  else:
    print(path,": ",gif.n_frames," 사용가능")
    
def divide(arr):
  evens=arr[0::2]
  odds=arr[1::2]
  if evens.shape is not odds.shape:
    evens=evens[0:-1]
  return [evens,odds]


arr=loadGif(path)