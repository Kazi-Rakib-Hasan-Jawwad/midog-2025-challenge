import os
from pathlib import Path
import numpy as np
from PIL import Image



files = os.listdir("/home/skim/MIDOG/test/test2/mask") # only for folder3, folder 4, folder 5  OK??

dir = os.getcwd()
a = 1

for i in files:
    j = os.path.join('/home/skim/MIDOG/test/test2/mask', i) # only for folder3, folder 4, folder 5   OK??
    a = Image.open(j)
    #print(i)
    #name = Path(j).stem
    dst = str(i)
    img = np.array(a)
    w, h = a.size
    np.place(img, img / 3 == 1, 1) # class 3 -> class 1
    np.place(img, img / 4 == 1, 2) # class 4 -> class 2
    img = Image.fromarray(np.array(img, dtype = 'uint8'))
    img.save(
        '/home/skim/MIDOG/test/test2/merged_mask/%s' % dst)  # only apply this code for folder 3, 4, 5 ok??
