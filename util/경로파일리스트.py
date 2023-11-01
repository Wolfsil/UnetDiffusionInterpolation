import os

path="/content/drive/MyDrive/gif이미지"
def getFilePath(path,end=""):
  gifFileList=os.listdir(path)
  gifPath=[]
  for name in gifFileList:
    if name.endswith(".gif"):
      gifPath.append(os.path.join(path,name))
  return gifPath
gifPath=getFilePath(path)