import os
from pathlib import Path
import numpy as np
from PIL import Image

files = os.listdir("/home/rakib/data/MIDOGpp-main/256/data/mask")
dir = os.getcwd()
a = 1

for i in files:
    j = os.path.join('/home/rakib/data/MIDOGpp-main/256/data/img', i)
    k = os.path.join('/home/rakib/data/MIDOGpp-main/256/data/mask', i)
    a = Image.open(j)
    b = Image.open(k)
    name = Path(j).stem
    dst = str(name) + '.png'
    w, h = a.size
    c = np.array(b)
    n0 = np.count_nonzero(c == 0)
    n1 = np.count_nonzero(c == 1)
    n2 = np.count_nonzero(c == 2)

    if n2 or n1 != 0:
        a.save('/home/rakib/data/MIDOGpp-main/256/cropped/img/%s' % dst)
        b.save('/home/rakib/data/MIDOGpp-main/256/cropped/mask/%s' % dst)
print("complete")