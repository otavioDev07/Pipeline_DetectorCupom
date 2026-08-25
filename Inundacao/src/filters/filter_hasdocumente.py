import cv2
import numpy as np
from draw import draw_line, draw_point


'''
NOME DO ARQUIVO DE ENTRADA:/input
NOME DO ARQUVIO DE SAIDA:/output


inserir os arquivos na pasta input nomeados por tags
0000.jpg, 0001.jpg, 0002.jpg

para utilizar basta criar um objeto passando a imagem:

        filtro = Filter_Hasdocument(img)

Então chamar a função 'do' do filtro:

        filtro.do()
'''


index = 1


class Filter_Hasdocument:


        def __init__(self , img):
                self.img_base = img
                self.img = img.copy()
                self.gray = None
                self.erodita = None
                self.dilatada = None
                #self.flooded = None
                self.edges = None
                self.result = None
                

        def do(self):
                
                lines = self.img2hough(self.img)
                points = [self.get_points(line) for line in lines] # points = [ x1, y1, x2, y2]
                
                i,j = 0, 1
                tam = len(lines)
                accept = [0] * tam

                while(i < tam):
                        # print("tamanho: ", tam)
                        # print("i: ", i)
                        # print("j: ", j)
                        # print(accept)
                        if(self.produto_interno_bruto(points[i], points[j])):
                                draw_line(points[i], self.img, color=(0,255,0))
                                draw_line(points[j], self.img, color=(0,255,0))
                                accept[i], accept[j] = 1, 1
                                #draw_point(self.find_intercession(points[i], points[j]),self.img,color=(255,0,0))

                                x, y = self.find_intercession(points[i], points[j])
                                x = int(x)
                                y = int(y)

                                cv2.circle(self.img, (x,y), 30, (255,0,0), -1)   

                        
                        j+=1
                        if(j == tam):
                                i+=1
                                if(i == tam-1):
                                        break
                                j=i+1


                i = 0
                while(i < tam):
                        if accept[i] == 0:
                               draw_line(points[i], self.img, color=(0,0,255))
                        i+=1

                self.drow_debug()

        def img2hough(self, img):
                self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                kernel = np.ones((5, 5), np.uint8)

                self.dilatada = cv2.dilate(self.gray, kernel, 30)

                # self.flooded = self.dilatada.copy()
    
                # h, w = self.flooded.shape[:2]
                # mask = np.zeros((h + 2, w + 2), np.uint8)
                
                # #seed_point = ( w // 2, h // 2)
                
                # cv2.floodFill(
                #         self.flooded, 
                #         mask, 
                #         seedPoint=(0,0), 
                #         newVal=0, 
                #         loDiff=3, 
                #         upDiff=3, 
                #         flags=8# Conectividade de 4 vizinhos
                # )


                self.erodida = cv2.erode(self.dilatada, kernel, 1)

                self.edges = cv2.Canny(self.erodida,50,150,apertureSize = 3)

                lines = cv2.HoughLines(self.edges,1,np.pi/180,100)

                return lines[:4]

        def get_points(self, line):
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


        def produto_interno_bruto(self, points1, points2, trashold=0.3):

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
                

        def find_intercession(self, line1, line2):
                
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
                
        def drow_debug(self):
                dilatada_bgr = cv2.cvtColor(self.dilatada, cv2.COLOR_GRAY2BGR)
                erodida_bgr  = cv2.cvtColor(self.erodida, cv2.COLOR_GRAY2BGR)
                edges_bgr    = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
                #flooded_bgr    = cv2.cvtColor(self.flooded, cv2.COLOR_GRAY2BGR)

                allOutputs = np.hstack((dilatada_bgr , erodida_bgr, edges_bgr, self.img, self.img_base))
                masterOutput = np.hstack((self.img, self.img_base))


                cv2.imwrite(f'output/{index:04}_alloutputs.png', allOutputs)
                cv2.imwrite(f'output/{index:04}.png', masterOutput)


        

while(index<=4000):
        img = cv2.imread(f'input/{index:04}.jpg')


        try:
                filtro = Filter_Hasdocument(img)

       

                print(f"processando {index}")

                filtro.do()

                print(f"finalizado {index}")

        except Exception as e:
                print("ERROR: " + str(e))
                
        index+=1

