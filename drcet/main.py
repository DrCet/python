import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt


model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
detector = hub.load(model_url)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512))
    return img, img_resized

original_image, input_image = load_image("page30.jpg")
input_tensor = tf.convert_to_tensor(input_image, dtype=tf.uint8)
input_tensor = input_tensor[tf.newaxis, ...]

detections = detector(input_tensor)

boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
threshold = 0.5
filtered_boxes = boxes[scores > threshold]
def scale_boxes(boxes, original_shape):
    height, width, _ = original_shape
    scaled_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        scaled_boxes.append([
            int(ymin * height), int(xmin * width),
            int(ymax * height), int(xmax * width)
        ])
    return np.array(scaled_boxes)

scaled_boxes = scale_boxes(filtered_boxes, original_image.shape)
def draw_boxes(image, boxes):
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

draw_boxes(original_image, scaled_boxes)
