import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from drcet.data.imagePreprocess import line_segment, paragraph_segment
import cv2

import matplotlib.pyplot as plt
import cv2
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = 'images\page132.jpg'

img, bboxes = paragraph_segment(path)
paragraphs = []

for bbox in bboxes:
    x,y,w,h = bbox
    if w>32 and h>32:
        cropped = img[y-4:y+h+4, x-4:x+w+4]
        paragraphs.append(cropped)

def extract_text(img):
    padded = cv2.copyMakeBorder(
        img,
        8,8,8,8,
        cv2.BORDER_CONSTANT,
        value = (255,255,255)
    )
    img, bboxes = line_segment(image = padded, scale = (1,1))    # the paragraph was scaled previously
    par_text = ''
    
    for bbox in bboxes:
        x,y,w,h  = bbox
        if h > 16 and w > 32 and w > h:
            # cv2.rectangle(padded, (x-4,y-4), (x+w+4, y+h+4), (127,0,0), 2)
            cropped = img[y-10:y+h+6, x-10:x+w+6]
            text = pytesseract.image_to_string(cropped, config='--psm 6', lang ='umh')
            print(text)
            plt.imshow(cropped)
            plt.axis('off')
            plt.show()
            par_text += text
    return img, par_text

page_text = ''
for paragraph in paragraphs:
    img, par_text= extract_text(paragraph)
    # print(par_text)
    page_text += par_text
    plt.imshow(paragraph)
    plt.axis('off')
    plt.show()
print(page_text)
plt.imshow(img)
plt.axis('off')
plt.show()




# plot bounding boxes
# plt.imshow(img)
# plt.axis('off')
# plt.show()