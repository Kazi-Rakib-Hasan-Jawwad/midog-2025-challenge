import numpy as np
import cv2, zlib, base64, io
from PIL import Image
import json
import matplotlib.pyplot as plt
import pandas as pd

a = 1
#df = pd.read_excel('/home/skim/Book1.xlsx', sheet_name='Sheet1', engine='openpyxl')
with open(
        '/home/skim/Downloads/MIDOG.json') as json_file:
    labels = json.load(json_file)


    for i in range(30524):
        tumor_type = labels['images'][i]['tumor_type']
        id = labels['images'][i]['id']
        file_name = labels['images'][i]['file_name']
        height = labels['images'][i]['height']
        width = labels['images'][i]['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        dst = str(file_name)
        #dst = dst.replace('./images/', '')
        #len = (len(labels['annotations']))
        #print(len)

        for j in range(len(labels['annotations'])):
            image_id = labels['annotations'][j]['image_id']

            if image_id == id:
                x1, y1, x2, y2 = labels['annotations'][j]['bbox']
                classes = labels['annotations'][j]['category_id']

                polygon1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                print(polygon1.shape)

                cv2.fillPoly(mask, [polygon1], color = classes)
                cv2.imwrite(
                    '/home/skim/MIDOG/masks/%s' % (dst), mask)
                '''
                col = col + 1
                cv2.fillPoly(mask, [polygon1], 1)
                cv2.imwrite(
                    '/home/skim/detection/pixel_12/%s' % (dst), mask)
                '''
        '''
        cv2.imwrite(
            '/home/skim/detection/all_masks/%s' % (dst), mask)
        '''

