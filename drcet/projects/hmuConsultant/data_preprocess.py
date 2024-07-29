import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import tensorflow as tf
from drcet.research.imageClassification.mobileNetV2 import MobileNetV2
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

with open('projects/hmuConsultant/paragraph_classification_dataset/annotations.json', 'r') as f:
    data = json.load(f)

train_data = [example for example in data['train'] if example['label'] in ['0', '1']]
test_data = [example for example in data['test'] if example['label'] in ['0', '1']]

print(f'Number of train data: {len(train_data)}')
print(f'Number of test(valid) data: {len(test_data)}')

id2label = {'0': 'invalid', '1': 'valid'}
label2id = {'invalid': '0', 'valid': '1'}

model = MobileNetV2(n_class = 2, input_shape = (224,512,3))
model.summary()

image_dir = 'projects/hmuConsultant/paragraph_classification_dataset/images/'
heights = []
widths = []
desired_height = 224
desired_width = 512

processed_data = {'train': [], 'test': []}
for i, example in enumerate(train_data):
    path = example['image']
    try:
        img = cv2.imread(path)
        height = img.shape[0]
        width = img.shape[1]
        widths.append(width)
        heights.append(height)

        pad_vert_up = max(0, (desired_height - height) // 2)
        pad_vert_down = max(0, desired_height - height - pad_vert_up)
        pad_horiz_left = max(0, (desired_width - width) // 2)
        pad_horiz_right = max(0, desired_width - width - pad_horiz_left)

        
        padded_image = cv2.copyMakeBorder(img, pad_vert_up, pad_vert_down, pad_horiz_left, pad_horiz_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if padded_image.shape[0] > desired_height or padded_image.shape[1] > desired_width:
            padded_image = cv2.resize(padded_image, (desired_width, desired_height))

        save_path = f'projects/hmuConsultant/paragraph_classification_dataset/image_preprocessed/train/{i}.jpg'
        label = example['label']
        cv2.imwrite(save_path, padded_image)
        entry = {'image': save_path,
                 'label': label}
        processed_data['train'].append(entry)
        print(f'Shape after padding: {padded_image.shape}')
    except:
        pass

for i, example in enumerate(test_data):
    path = example['image']
    try:
        img = cv2.imread(path)
        height = img.shape[0]
        width = img.shape[1]
        widths.append(width)
        heights.append(height)

        pad_vert_up = max(0, (desired_height - height) // 2)
        pad_vert_down = max(0, desired_height - height - pad_vert_up)
        pad_horiz_left = max(0, (desired_width - width) // 2)
        pad_horiz_right = max(0, desired_width - width - pad_horiz_left)

        
        padded_image = cv2.copyMakeBorder(img, pad_vert_up, pad_vert_down, pad_horiz_left, pad_horiz_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if padded_image.shape[0] > desired_height or padded_image.shape[1] > desired_width:
            padded_image = cv2.resize(padded_image, (desired_width, desired_height))

        save_path = f'projects/hmuConsultant/paragraph_classification_dataset/image_preprocessed/test/{i}.jpg'
        label = example['label']
        cv2.imwrite(save_path, padded_image)
        entry = {'image': save_path,
                 'label': label}
        processed_data['test'].append(entry)
        print(f'Shape after padding: {padded_image.shape}')
    except:
        pass
with open('projects/hmuConsultant/paragraph_classification_dataset/image_preprocessed/annotations.json', 'w') as f:
    json.dump(processed_data, f, indent = 4)
