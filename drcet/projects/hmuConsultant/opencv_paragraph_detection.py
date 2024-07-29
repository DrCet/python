import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import cv2
from drcet.data.imagePreprocess import character_segment, line_segment, paragraph_segment
import matplotlib.pyplot as plt
import tensorflow as tf


path = 'images\page17.jpg'

img, bboxes = paragraph_segment(path)

for bbox in bboxes:
    x,y,w,h = bbox
    if w>32 and h>32:
        cv2.rectangle(img, (x-4, y-4), (x+w+4, y+h+4), (0, 255, 0), 2)


plt.imshow(img)
plt.axis('off')
plt.show()
