import cv2
import numpy as np

class Operator:

    def __init__(self, img):
        self.img = img
        self.canny = None

    def toCanny(self, min=60, max=130):
        self.canny = cv2.Canny(self.img,min,max,apertureSize = 3)
        return self.canny

    def makeHough_h(self, min=90, max=100):
        min_theta = np.radians(min)
        max_theta=np.radians(max)
        return cv2.HoughLines(self.canny,1,np.pi/180,25, min_theta=min_theta, max_theta=max_theta, srn=3, stn=0)
    
    def makeHough_v(self, min=-10, max=5):
        min_theta = np.radians(min)
        max_theta = np.radians(max)
        return cv2.HoughLines(self.canny,1,np.pi/180,25, min_theta=min_theta, max_theta=max_theta, srn=3, stn=0)

    