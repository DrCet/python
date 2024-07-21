import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from drcet.data.imagePreprocess import line_segment
import cv2
save_dir = 'umh-ground-truth'
def crop_img(path):
    img, bboxes = line_segment(path, kernel = (1,80), scale = (2,2))
    i = 0
    count = 0
    for bbox in bboxes:
        x,y,w,h = bbox
        if h>16 and w>32:
            cropped = img[y-4:y+h+4, x-4:x+w+4]
            i += 1
            save_path = os.path.join(save_dir, f'{path[:-4]}__{i}.png')
            cv2.imwrite(save_path, cropped)
            print(save_path)
            count += 1
    return count
total_example = 0
for i in range(368,374):
    path = f"page{i}.jpg"
    c = crop_img(path)
    total_example += c
for i in range(127,135):
    path = f"page{i}.jpg"
    c = crop_img(path)
    total_example += c

print(total_example)
