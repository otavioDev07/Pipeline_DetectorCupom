import numpy as np

class Finder:

    def __init__(self):
        pass

    def isPerpendicular(self, points1, points2, trashold=0.3):
            vector1 = [points1[0] - points1[2], points1[1] - points1[3]]
            vector2 = [points2[0] - points2[2], points2[1] - points2[3]]
            modulo1 = np.linalg.norm(vector1)
            modulo2 = np.linalg.norm(vector2)
            
            a = (vector1[0]/modulo1) * (vector2[0]/modulo2)
            b = (vector1[1]/modulo1) * (vector2[1]/modulo2)
            c = a + b
            if abs(c) <= trashold:
                    return True
            else:
                    return False
            
    def intercession(self, line1, line2):
            
            denominador_m1 = (line1[0] - line1[2])
            if denominador_m1 == 0: 
                    denominador_m1 = 0.0001 
                    
            denominador_m2 = (line2[0] - line2[2])
            if denominador_m2 == 0: 
                    denominador_m2 = 0.0001 
            m1 = (line1[1] - line1[3]) / denominador_m1
            m2 = (line2[1] - line2[3]) / denominador_m2
            a1 = m1
            a2 = m2
            b1 = (m1 * (-line1[0])) + line1[1]
            b2 = (m2 * (-line2[0])) + line2[1]
            #nescessariamente, a1 - a2 nao podem ser 0, pois l1 e l2 formam 90 graus
            x_interc = (b2 - b1) / (a1 - a2)
            y_interc = (x_interc * a1)+ b1
            return x_interc, y_interc

    def isSquare(self, points, altura, largura):
        accept = [0] * 4
        middle_y, middle_x = altura // 2, largura // 2
        point_1 = []
        point_2 = []
        point_3 = []
        point_4 = []

        for point in points:

                if(point[0] < middle_x and point[1] < middle_y):
                        point_1.append(point)
                        accept[0] = 1
                elif(point[0] > middle_x and point[1] < middle_y):
                        point_2.append(point)
                        accept[1] = 1
                elif(point[0] > middle_x and point[1] > middle_y):
                        point_3.append(point)
                        accept[2] = 1
                else:
                        point_4.append(point)
                        accept[3] = 1

        for ac in accept:
                if ac == 1:
                        continue
                return 0, None, None, None, None              

        return 1, self.get_middle_point(point_1), self.get_middle_point(point_2), self.get_middle_point(point_3), self.get_middle_point(point_4)

    def get_middle_point(self, coordenadas):
            point = [0,0]
            n = len(coordenadas)
            # Somando os elementos
            soma_x = sum(p[0] for p in coordenadas)
            soma_y = sum(p[1] for p in coordenadas)
            # Calculando a média
            point[0] = soma_x / n
            point[1] = soma_y / n
            return point