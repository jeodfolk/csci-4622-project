import cv2
import numpy as np


def crop(img):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    background = [255, 0, 255, 0]
    rows,cols,_ = image.shape
    top = None
    for i in range(rows):
        for j in range(cols):
            if (image[i,j] != background).all(): 
                # print("top", [i,j])
                top = i
                break
        if top: break
    
    bottom = None
    for i in reversed(range(rows)):
        for j in range(cols):
            if (image[i,j] != background).all(): 
                # print("bottom", [i,j])
                bottom = i
                break
        if bottom: break

    left = None
    for j in range(cols):
        for i in range(rows):
            if (image[i,j] != background).all(): 
                # print("left", [i,j])
                left = j
                break
        if left: break
    
    right = None
    for j in reversed(range(cols)):
        for i in range(rows):
            if (image[i,j] != background).all(): 
                # print("right", [i,j])
                right = j
                break
        if right: break

    # cv2.rectangle(image, (left, top), (right, bottom), (36,255,12), 1)
    # cv2.imshow('image', image)
    # cv2.imshow('i', image[top:bottom, left:right])
    # cv2.waitKey()
    return left, bottom, right, top
