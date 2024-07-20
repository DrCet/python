import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from drcet.data.imagePreprocess import line_segment
import cv2

import matplotlib.pyplot as plt
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(path):
    img, bboxes = line_segment(path, kernel = (1,80), scale = (2,2))    
    full_text = ''
    for bbox in bboxes:
        x,y,w,h  = bbox
        if h > 16 and w > 32:
            # cv2.rectangle(img, (x-4,y-4), (x+w+4, y+h+4), (127,0,0), 2)
            cropped = img[y-4:y+h+4, x-4:x+w+4]
            text = pytesseract.image_to_string(cropped, config='--psm 6', lang ='hmu')
            full_text += text
    return img, full_text
path = 'page373.jpg'
img, full_text = extract_text(path)
print(full_text)
# plot bounding boxes
# plt.imshow(img)
# plt.axis('off')
# plt.show()
