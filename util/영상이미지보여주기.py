import matplotlib.pyplot as plt
import numpy as np
#분리한 이미지를 천천히 한장씩 보여주는 코드
from IPython import display
from PIL import Image

path=""
gif=Image.open(path)

images=[]
for i in range(1,gif.n_frames):
    gif.seek(i)
    temp=gif.convert("RGBA")
    images.append(np.array(temp))
gif.close()
    
for i in images:
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.imshow(i)
