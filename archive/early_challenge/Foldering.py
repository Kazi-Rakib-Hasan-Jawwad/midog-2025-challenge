import os
from pathlib import Path
import numpy as np
from PIL import Image



files = os.listdir("/home/skim/MIDOG/test/test2/mask")

dir = os.getcwd()
a = 1

for i in files:
    img_path = os.path.join("/home/skim/MIDOG/test/test2/img", i)
    mask_path = os.path.join("/home/skim/MIDOG/test/test2/mask", i)

    img = Image.open(img_path)
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    name = Path(img_path).stem
    dst = f"{name}.png"


    n0 = np.count_nonzero(mask_np == 0)
    n1 = np.count_nonzero(mask_np == 1)
    n2 = np.count_nonzero(mask_np == 2)
    n3 = np.count_nonzero(mask_np == 3)
    n4 = np.count_nonzero(mask_np == 4)


    unique_classes = set(np.unique(mask_np)) - {0}

    if not unique_classes:
        continue
    elif unique_classes == {1}:
        img.save('/folder1/img/%s' % dst)
        mask.save('/folder1/mask/%s' % dst)
    elif unique_classes == {2}:
        img.save('/folder2/img/%s' % dst)
        mask.save('/folder2/mask/%s' % dst)
    elif unique_classes.issubset({1, 3}) and unique_classes:
        img.save('/folder3/img/%s' % dst)
        mask.save('/folder3/mask/%s' % dst)
    elif unique_classes.issubset({2, 4}) and unique_classes:
        img.save('/folder4/img/%s' % dst)
        mask.save('/folder4/mask/%s' % dst)
    else:
        img.save('/folder5/img/%s' % dst)
        mask.save('/folder5/mask/%s' % dst)