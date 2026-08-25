import cv2


def draw_line(line, img, color):
        x1,y1,x2,y2 = line
        
        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def draw_point(point, img, color):
        cv2.circle(img, (int(point[0]), int(point[1])), 20, color, -1)    
        cv2.imwrite('Janela.jpg', img)  
        print(f"ponto ({point[0]},{point[1]}) desenhado")