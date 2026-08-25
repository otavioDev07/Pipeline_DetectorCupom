import statistics
import numpy as np
from tools_filter import normalize

trashold = 0.235

class Classifier:

    def __init__(self):
        pass

    
    def vector_binary_selection(self, vector_fft_factor):
        '''
        RECEBE UM VETOR DE VALORES E CLASSIFICA EM DOIS GRUPOS COM TRASHOLD DINAMICO
        RETORNA UM VETOR DE 0's E 1's QUE CLASSIFICA RESPECTIVAMENTE OS VALORES
        DE ENTRADA
        '''
        
        len_vector = len(vector_fft_factor) # Pega tamanho do vetor

        vector_selectioned = [0] * len_vector # Inicializa vetor de selecao

        mean = statistics.mean(vector_fft_factor) # Pega a media das medias das alto frequencias

        '''
        ACEITA TODAS AS IMAGENS QUE ESTÃO ACIMA DA MEDIA
        '''
        i = 0 
        while(i < len_vector):

            if(vector_fft_factor[i] > mean): # trashold dinamico 1 :: valores abaixo da media
                vector_selectioned[i] = 1

            i+=1

        #Desvio padrao
        stndev = statistics.pstdev(vector_fft_factor) # Pega o desvio padrao dos dados

        '''
        ACEITA TODAS AS IMAGENS QUE ESTAO A 1 DESVIO PADRAO DE DISTANCIA DA MEDIA
        '''
        i = 0 
        while ( i < len_vector):
            if(vector_selectioned[i] == 0): # trashold dinamico 2 :: valores que estao a menos de -
                distance = np.abs((vector_fft_factor[i] - mean)) # - 1 desvio padrao de distancia da media
                print(distance)
                if(distance <= stndev):
                    vector_selectioned[i] = 1

            i += 1

        return vector_selectioned

    def unit_binary_selection(self, img_factor, trashold):
        '''
        RECEBE O FATOR FFT DE UMA IMAGEM E O TRASHOLD
        DEVOLVE 0 CASO ELA ESTEJA ABAIXO DO TRASHOLD
        DEVOLVE 1 CASO ELA ESTEJA ACIMA DO TRASHOLD
        '''
        if(img_factor > trashold):
            return 1
        else:   
            return 0

    def rank_selection(self, vector_fft_factor):
        '''
        RECEBE UM VETOR DE VALORES E CLASSIFICA CLASSIFICA OS VALORES EM NOTAS DE [0--1]
        ONDE 1 É A MELHOR NOTA DO VETOR E 0 É A PIOR
        '''
        vector_normalized = normalize(vector_fft_factor)

        return vector_normalized * 10
    







