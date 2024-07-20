import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from drcet.data.imagePreprocess import line_segment
import numpy as np
import cv2
import matplotlib.pyplot as plt


path = 'page30.jpg'
img, bboxes = line_segment(path, kernel = (1,100))
for bbox in bboxes:
    x, y, w, h = bbox
    if w> 10 and h>10:
        cv2.rectangle(img, (x, y), (x+w, y+h), (127,0,0), 2)

plt.imshow(img)
plt.axis('off')
plt.show()