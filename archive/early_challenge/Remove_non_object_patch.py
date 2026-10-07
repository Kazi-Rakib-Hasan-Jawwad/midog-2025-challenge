import os
from pathlib import Path
import numpy as np
from PIL import Image

files = os.listdir("/home/rakib/data/MIDOGpp-main/cropped/img")

dir = os.getcwd()
a = 1
# 이미지 사이즈랑 픽셀 별 클래스 개수

for i in files:
    j = os.path.join('/home/rakib/data/MIDOGpp-main/cropped/img', i)
    k = os.path.join('/home/rakib/data/MIDOGpp-main/cropped/mask', i)
    a = Image.open(j)
    b = Image.open(k)
    #print(i)
    c = np.array(b)
    name = Path(j).stem
    dst = str(name) + '.png'
    w, h = a.size
    n0 = np.count_nonzero(c == 0)
    n1 = np.count_nonzero(c == 1)
    n2 = np.count_nonzero(c == 2)


    if n2 or n1 != 0:

        a.save('/home/rakib/data/MIDOGpp-main/cropped2/img/%s' % dst)
        b.save('/home/rakib/data/MIDOGpp-main/cropped2/mask/%s' % dst)

print('working!')