import cv2
import numpy as np
from draw import draw_line, draw_point
from image import Image
from operator_ import Operator
from lines import Lines
from finder import Finder


DEBUG = True

class HasDocument:

    def __init__(self, img):
        self.img = img
        self.result = img.copy()
     
    def do(self):
        flag = 0
        p1, p2, p3 , p4 = None, None, None, None

        image = Image(self.img)
        finder = Finder()
        

        image.toGray()
        image.toDilate(image.g)
        image.toErose(image.dilate)
 
        operator = Operator(img=image.erode)
        
        lines = Lines()

        max_tentativas = 3
        t = 0

        while(t < max_tentativas):

            operator.toCanny(60 - (t * 20), 130 - (t * 20))

            lines.h = operator.makeHough_h(90 - (t * 6), 100 + (t * 6))
            lines.v = operator.makeHough_v(-10 - (t * 6), 5 + (t * 6))

            lines.points_h = lines.toPoints(lines.h, 4)
            lines.points_v = lines.toPoints(lines.v, 4)

            tam_h = len(lines.points_h)
            tam_v = len(lines.points_v)
            i,j = tam_h,0
            
            accept = [0] * (tam_h + tam_v)
            points_intercection = []

            for h in lines.points_h:
                    for v in lines.points_v:
                            if(finder.isPerpendicular(h, v)):
                                    if(DEBUG):
                                        draw_line(h, self.result, color=(0,255,0))
                                        draw_line(v, self.result, color=(0,255,0))
                                    accept[j], accept[i] = 1, 1
                                    x, y = finder.intercession(h, v)
                                    
                                    x = int(x)
                                    y = int(y)
                                    points_intercection.append((x,y))
                                    if(DEBUG):
                                        cv2.circle(self.result, (x,y), 30, (255,0,0), -1) 
                            i+=1
                    j+=1
                    i=tam_h
            if(DEBUG):
                i = 0
                while(i < tam_h):
                        if accept[i] == 0:
                                draw_line(lines.points_h[i], self.result, color=(0,0,255))
                        i+=1
                while(i < (tam_h + tam_v - 1)):
                        if accept[i] == 0:
                                draw_line(lines.points_v[i - tam_h + 1], self.result, color=(0,0,255))
                        i+=1

            

            flag, p1, p2, p3, p4 = finder.isSquare(points=points_intercection,altura=self.img.shape[0],largura=img.shape[1])


            if (flag == 1):
                    if(DEBUG):
                        cv2.circle(self.result, (int(p1[0]), int(p1[1])) , 30, (0,0,255), -1)                        
                        cv2.circle(self.result, (int(p2[0]), int(p2[1])) , 30, (0,0,255), -1)
                        cv2.circle(self.result, (int(p3[0]), int(p3[1])) , 30, (0,0,255), -1)
                        cv2.circle(self.result, (int(p4[0]), int(p4[1])) , 30, (0,0,255), -1)
                    break

            t+=1

        return flag, p1, p2, p3, p4    #se achou e onde achou


caminho_imagem = 'filter_has_document/img_teste.jpg'
img = cv2.imread(caminho_imagem)

filtro = HasDocument(img)
flag, p1, p2, p3, p4 = filtro.do()
print(flag, p1, p2, p3, p4)

cv2.imwrite('filter_has_document/output.jpg', filtro.result)
