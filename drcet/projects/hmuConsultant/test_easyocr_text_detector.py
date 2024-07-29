
# this aprroach didn't work well >>> attempt to use opencv only

import easyocr
import cv2
import matplotlib.pyplot as plt

reader = easyocr.Reader(['vi'])
image_path ='images/page49.jpg'
image = cv2.imread(image_path)
results = reader.readtext(image_path)


paragraph = []
n_paragraphs = 0
boxes = []

for (bbox, txt, prob) in results:
    boxes.append(bbox)
boxes = sorted(boxes, key = lambda bbox: bbox[0][1])

for bbox in boxes:
    (top_left, top_right,bottom_right, bottom_left) = bbox

    # print(f'Width of this box: {abs(top_left[0] - top_right[0])}')
    # print(f'Height of this box: {abs(top_left[1] - bottom_left[1])}')
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    paragraph.append(bbox)

    if len(paragraph) > 1:
        cur_top_left_x = bbox[0][0]
        cur_top_left_y = bbox[0][1]
        
        prev_top_right_x = paragraph[-2][1][0]
        prev_bottom_left_y = paragraph[-2][3][1]
        prev_top_left_x = paragraph[-2][0][0]
        prev_top_left_y = paragraph[-2][0][1]

        prev_height = abs(prev_top_left_y - prev_bottom_left_y)
        prev_width = abs(prev_top_left_x - prev_top_right_x)
        x_distance = abs(abs(cur_top_left_x - prev_top_left_x) - prev_width)
        y_distance = abs(abs(cur_top_left_y - prev_top_left_y) - prev_height) 

        max_y_distance = 16
        max_x_distance = 8

        if x_distance > max_x_distance and y_distance > max_y_distance:
            min_x = min(box[0][0] for box in paragraph)
            min_y = min(box[0][1] for box in paragraph)
            max_x = max(box[2][0] for box in paragraph)
            max_y = max(box[2][1] for box in paragraph)

            par_top_left = (int(min_x), int(min_y))
            par_bottom_right = (int(max_x), int(max_y))
            try:
                cv2.rectangle(image, par_top_left, par_bottom_right, (0, 255, 0), 2)
                print('Successful drawed this box:', par_top_left, par_bottom_right)
            except:
                print('An error occurred when trying to draw this box',par_top_left, par_bottom_right)
            n_paragraphs += 1
            print(f'Drawed {n_paragraphs} paragraphs')
            paragraph = [bbox]


plt.imshow(image)
plt.axis('off')
plt.show()

'''# Initialize EasyOCR reader with Vietnamese language
reader = easyocr.Reader(['vi'])

# Load image
image_path = 'images/page49.jpg'
image = cv2.imread(image_path)

# Perform text detection
results = reader.readtext(image_path)

# Draw bounding boxes
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    print(top_left,'---', bottom_right)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Display the result
plt.imshow(image)
plt.axis('off')
plt.show()
'''