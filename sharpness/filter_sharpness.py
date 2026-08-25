import cv2
import numpy as np
from tools_filter import normalize

class Filter_Sharpness:

    def __init__(self):
        self.mask_percentage = 0.6
    
    def do(self, img, mask_percentage=None):
        '''
        RECEBE UMA IMAGEM E A PORCENTAGEM DE CORTE QUE SERA UTILIZADO PARA A IMAGEM NO ESPECTRO DAS ALTO FREQUENCIAS
        E RETORNA O FATOR DE ALTO FREQUENCIA DESSA IMAGEM
        '''

        if(mask_percentage == None):
            mask_percentage = self.mask_percentage

        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        img_gray_to_fft = img_gray.astype(np.float32) / 255.0

        V_magnetitude_spectrum = self.img_to_magnetitude_spectrum(img_gray_to_fft, mask_percentage)

        V_magnetitude_spectrum_normalized = normalize(V_magnetitude_spectrum)

        mean_V_magnetitude_spectrum_normalized = np.mean(V_magnetitude_spectrum_normalized)

        return float(mean_V_magnetitude_spectrum_normalized)
    

    def img_to_frequence(self, img):

        f = np.fft.fft2(img)

        return np.fft.fftshift(f)
    

    def img_to_magnetitude_spectrum(self, img, mask_percentage):

        fshift = self.img_to_frequence(np.array(img))

        height, width = fshift.shape

        center_x, center_y = width // 2, height // 2

        center_height = int(height * mask_percentage)
        center_width = int(width * mask_percentage)
        x_min, x_max = center_x - (center_width // 2), center_x + (center_width // 2)
        y_min, y_max = center_y - (center_height // 2), center_y + (center_height // 2)

        f_cutted = fshift[y_min:y_max, x_min:x_max]

        magnetitude_spectrum = np.log(1+np.abs(f_cutted)) * 255 #Aqui o 1 + serve para não ocorrer log(0)
        
        return magnetitude_spectrum

