import numpy as np
import cv2

def line_segment(path, kernel: tuple = (5,80), scale: tuple = (2,2)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(0,0), fx = scale[0], fy = scale[1])
    
    org_img = cv2.imread(path)
    org_img = cv2.resize(org_img, (0,0), fx = scale[0], fy = scale[1])

    #img = cv2.GaussianBlur(img, (3,3), 0)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 1))
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    morph = cv2.erode(morph, kernel2, iterations=1)

    (contours, hierarchy) = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2. CHAIN_APPROX_SIMPLE)
    sorted_contour_lines = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[1])
    bboxes = []
    for ctr in sorted_contour_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        bboxes.append((x,y,w,h))
    return org_img, bboxes
