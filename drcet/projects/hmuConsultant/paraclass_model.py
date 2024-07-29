import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import tensorflow as tf
from drcet.research.imageClassification.mobileNetV2 import MobileNetV2
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

with open('projects/hmuConsultant/paragraph_classification_dataset/image_preprocessed/annotations.json', 'r') as f:
    data = json.load(f)

n_train = len(data['train'][:500])
n_test = len(data['test'][:100])

train_data = np.zeros((n_train, 224, 512, 3))
test_data = np.zeros((n_test, 224, 512, 3))

train_label = []
test_label = []

for i, example in enumerate(data['train'][:500]):
    path = example['image']
    img = cv2.imread(path)
    train_data[i] = img
    train_label.append(int(example['label']))

for i, example in enumerate(data['test'][:100]):
    path = example['image']
    img = cv2.imread(path)
    test_data[i] = img
    test_label.append(int(example['label']))

train_label = np.array(train_label)
test_label = np.array(test_label)

model = MobileNetV2(n_class = 2, input_shape = (224,512,3))
model.summary()
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
)

history = model.fit(train_data, train_label, epochs = 5, validation_data = (test_data, test_label))

plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label = 'Train loss')
plt.plot(history.history['val_loss'], label = 'Val loss')
plt.title('Model loss')
plt.xticks([1,2,3])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')


plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label = 'Train acc')
plt.plot(history.history['val_accuracy'], label = 'Val acc')
plt.title('Model acc')
plt.xticks([1,2,3])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend(loc='upper left')

plt.show()