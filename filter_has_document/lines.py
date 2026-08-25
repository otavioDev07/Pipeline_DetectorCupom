import cv2
import numpy as np

class Lines:

    def __init__(self):
        self.h = None
        self.v = None
        self.points_h = None
        self.points_v = None

    def toPoints(self, lines, numLines):
            points = [self.getPoints(line) for line in lines[:numLines]]
    
            return points[:numLines]
    
    def getPoints(self, line):
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            return x1,y1,x2,y2