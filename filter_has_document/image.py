import cv2
import numpy as np

class Image:

    def __init__(self, img):
        self.rgb = img
        self.g = None
        self.dilate = None
        self.erode = None

    def toGray(self):
        self.g = cv2.cvtColor(self.rgb,cv2.COLOR_BGR2GRAY)
        return self.g

    def toDilate(self, img):
        kernel = np.ones((8, 8), np.uint8)
        self.dilate = cv2.dilate(img, kernel, 11)
        return self.dilate

    def toErose(self, img):
        kernel = np.ones((8, 8), np.uint8)
        self.erode = cv2.erode(img, kernel, 11)
        return self.erode


