# %% [markdown]
# # incluindo bibliotecas

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

from src.filters.base_filter import BaseFilter
from src.log.log_body import LogBody


# %% [markdown]
# # Setup de Globais
# * setup do diretório de input
# * setup do diretório de output

# %%
#input = "./input2/"
input = "./input/"
output = "./output/"
fname = "" 

# %% [markdown]
# # declaração de funções necessárias
# * img_show
# * img_show_lado_a_lado

# %%
def img_show(img, titulo="Imagem 1", map='gray'):
    if img is None:
        print("img não foi carregada.")
        return None
    if len(getattr(img, 'shape', [])) == 2:
        plt.imshow(img, cmap=map)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")

def img_show_lado_a_lado(imgs, titles=None, figsize_per_image=(2,4), map='gray', save=False):
    """Exibe uma sequência de imagens lado a lado seguindo a ordem do vetor `imgs`."""
    if imgs is None:
        print("Nenhuma imagem fornecida.")
        return None
    imgs = list(imgs)
    n = len(imgs)
    if n == 0:
        print("Lista de imagens vazia.")
        return None
    if titles is None:
        titles = [f"Imagem {i+1}" for i in range(n)]
    else:
        titles = list(titles)
        if len(titles) < n:
            titles += [f"Imagem {i+1}" for i in range(len(titles), n)]
    max_per_row=4
    cols = min(n, max_per_row)
    rows = (n + cols - 1) // cols

    fig_w = figsize_per_image[0] * cols
    fig_h = figsize_per_image[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # normalizar axes para lista
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    for idx in range(rows * cols):
        ax = axes_flat[idx]
        if idx >= n:
            ax.axis('off')
            continue
        img = imgs[idx]
        title = titles[idx]
        if img is None:
            ax.text(0.5, 0.5, "None", ha='center', va='center')
            ax.set_title(title)
            ax.axis("off")
            continue
        if len(getattr(img, 'shape', [])) == 2:
            ax.imshow(img, cmap=map)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2, edgecolor='black', facecolor='none', zorder=10)
        ax.add_patch(rect)
    plt.tight_layout()
    if save:
        path = os.path.join(output, fname)
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

# %% 
# # Classe inundacao 

class FilterWaterShed(BaseFilter):

    def __init__(self):
        self.mask_percentage = 0.25
        self.max_poly_vertices = 6
        self.min_rect_extent = 0.7  # o quanto o polígono precisa preencher seu retângulo mínimo para "parecer" um retângulo

    @property
    def name(self) -> str:
        return 'watershed'

    def apply(self, path, log_body: LogBody):
        markers, inundacao = self.inundacao_document(path, show=False)
        acepted = self.accepted(markers, inundacao, save=False)
        log_body.filters_applied.append(self.name)
        log_body.accepted = acepted  # Watershed não rejeita a imagem, apenas segmenta. 
        #fazer detecção com base no tamanho do maior rótulo
        log_body.markers = markers
        return markers, inundacao
    

    def inundacao_document(self, img_path=None, show=False):
        """Processa a imagem para segmentação por watershed e retorna (markers, inundacao)."""
        if img_path is None:
            img_path = input + '0001.jpg'
        self.img = cv2.imread(img_path)

        self.escala = 0.5
        self.img = cv2.resize(self.img, None, fx=self.escala, fy=self.escala, interpolation=cv2.INTER_AREA)

        if self.img is None:
            print('Imagem não encontrada:', img_path)
            return None, None
    
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imgs_fg = [self.img, self.gray]
    
        # median blur remove ruído fino de textura do fundo antes da binarização, preservando bordas
        blurred = cv2.medianBlur(self.gray, 7)
        block_size = 45
        self.bin = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
        
        imgs_fg.append(self.bin)
    
        kernel = np.ones((3, 3), np.uint8)
        self.closed = cv2.morphologyEx(self.bin, cv2.MORPH_ERODE, kernel, iterations=2)
        self.closed = cv2.morphologyEx(self.closed, cv2.MORPH_CLOSE, kernel, iterations=2)

        imgs_fg.append(self.closed)
    
        num_labels, markers = cv2.connectedComponents(self.closed)
        if num_labels <= 1:
            # Nenhum componente de objeto foi encontrado; mantém foreground vazio.
            self.sure_fg_uint8 = np.zeros_like(self.closed, dtype=np.uint8)
        else:
            component_sizes = np.bincount(markers.ravel())
            max_label = np.argmax(component_sizes[1:]) + 1  # Ignorar o background (0)
            self.sure_fg_uint8 = (markers == max_label).astype(np.uint8) * 255

        self.sure_fg_uint8 = cv2.morphologyEx(self.sure_fg_uint8, cv2.MORPH_CLOSE, kernel, iterations=8)
        sure_fg_bin = self.sure_fg_uint8

        self.norm = self.sure_fg_uint8.copy()
        imgs_fg.append(self.norm)
    
        if show:
            img_show_lado_a_lado(imgs_fg, titles=['Original', 
                                                  'Cinza', 
                                                  'Binarizada', 
                                                  'Fechada', 
                                                  'Objeto'], map='gray')
    
        # O maior componente da região desconhecida é assumido como fundo.

        self.unknown = (sure_fg_bin == 0).astype(np.uint8)
        num_labels_bg, markers = cv2.connectedComponents(self.unknown)
        if num_labels_bg <= 1:
            self.sure_bg_uint8 = np.zeros_like(self.sure_fg_uint8, dtype=np.uint8)
        else:
            component_sizes_bg = np.bincount(markers.ravel())
            max_label_bg = np.argmax(component_sizes_bg[1:]) + 1
            self.sure_bg_uint8 = (markers == max_label_bg).astype(np.uint8) * 255
        self.sure_bg_uint8 = cv2.erode(self.sure_bg_uint8, kernel, iterations=20)
    
        fg_or_bg = cv2.bitwise_or(self.sure_fg_uint8, self.sure_bg_uint8)
        self.unknown = cv2.bitwise_not(fg_or_bg)
    
        if show:
            imgs_bg = [self.img, self.sure_fg_uint8, self.sure_bg_uint8, self.unknown]
            img_show_lado_a_lado(imgs_bg, titles=['Original', 
                                                  'Objeto', 
                                                  'Fundo', 
                                                  'Desconhecido'], map='gray')
    
        # rotular componentes do foreground e preparar marcadores
        markers = cv2.connectedComponents(self.sure_fg_uint8)[1].astype(np.int32)
    
        # marcar regiões desconhecidas com 0 (requisito do watershed)
        markers[self.unknown == 255] = 0
        # incrementar rótulos positivos para reservar 1 como background
        markers[markers > 0] += 1
        markers[self.sure_bg_uint8 == 255] = 1
    
        if show:
            print('Valores dos marcadores (amostra):', np.unique(markers)[:50])
    
        # aplicar o algoritmo de inundação
        inundacao = cv2.watershed(self.img, markers.copy())
        self.inundacao = inundacao
    
        if show:
            img_show_lado_a_lado([self.img, markers, inundacao], titles=['Original', 
                                                                    'Marcadores', 
                                                                    'Inundação'], map='jet')
    
        return markers, inundacao

    def accepted(self, markers, inundacao, save=False):
        """Verifica se o cupom fiscal foi encontrado e delimitado por um polígono válido."""
        labels, counts = np.unique(inundacao, return_counts=True)
        # Ignorar o rótulo -1 (bordas do watershed) e 1 (background)
        valid_labels = labels[(labels > 1) & (labels != -1)]

        poly_points = None
        if len(valid_labels) > 0:
            # Encontrar o rótulo com maior área
            max_label = valid_labels[np.argmax(counts[(labels > 1) & (labels != -1)])]
            max_count = counts[labels == max_label][0]
            total_pixels = inundacao.size
            
            # Se o maior rótulo ocupa mais de mask_percentage da imagem
            if max_count > self.mask_percentage * total_pixels:
                # Extrair contorno do rótulo máximo e aproximar por um polígono
                mask = (inundacao == max_label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    # o hull convexo remove reentrâncias/ruídos, aproximando o polígono de um retângulo
                    hull = cv2.convexHull(cnt)
                    perimeter = cv2.arcLength(hull, True)
                    epsilon_factor = 0.02
                    candidate = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
                    # aumenta o epsilon até o polígono caber no máximo de vértices predefinido
                    while len(candidate) > self.max_poly_vertices and epsilon_factor < 1.0:
                        epsilon_factor += 0.02
                        candidate = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

                    if 3 <= len(candidate) <= self.max_poly_vertices:
                        # verifica se o polígono se aproxima de um retângulo: área do polígono
                        # comparada à área do seu menor retângulo envolvente (minAreaRect)
                        rect_w, rect_h = cv2.minAreaRect(candidate)[1]
                        rect_area = rect_w * rect_h
                        poly_area = cv2.contourArea(candidate)
                        rect_extent = poly_area / rect_area if rect_area > 0 else 0
                        if rect_extent >= self.min_rect_extent:
                            poly_points = candidate

        # só aceitamos o cupom se conseguimos sinalizar sua área com um polígono válido e retangular
        cupom_encontrado = poly_points is not None

        # sinaliza no console se o cupom fiscal foi encontrado ou não
        if cupom_encontrado:
            print(f"Cupom fiscal encontrado na imagem de entrada (polígono com {len(poly_points)} vértices).")
        else:
            print("Cupom fiscal não encontrado na imagem de entrada.")

        # visualização com cores fixas (não depende do range de valores de inundacao): objeto em vermelho, fundo em amarelo
        self.inundacao = np.zeros((*inundacao.shape, 3), dtype=np.uint8)
        self.inundacao[inundacao == 1] = (0, 255, 255)  # fundo (amarelo, BGR)
        self.inundacao[inundacao > 1] = (0, 0, 255)     # objeto (vermelho, BGR)
        self.inundacao[inundacao == -1] = (255, 255, 255)  # bordas do watershed (branco)
        if cupom_encontrado:
            # desenha o polígono em azul, sem alterar as cores fixas de objeto/fundo
            cv2.polylines(self.inundacao, [poly_points], isClosed=True, color=(255, 0, 0), thickness=3)

        # painel de resultado: ocupa o espaço vazio do grid, logo após a Inundação
        status_text = "ACEITO" if cupom_encontrado else "REJEITADO"
        status_color = (0, 170, 0) if cupom_encontrado else (0, 0, 220)
        status_img = np.full_like(self.img, 255)
        cv2.rectangle(status_img, (0, 0), (status_img.shape[1] - 1, status_img.shape[0] - 1), status_color, thickness=12)
        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        text_x = (status_img.shape[1] - text_w) // 2
        text_y = (status_img.shape[0] + text_h) // 2
        cv2.putText(status_img, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3, cv2.LINE_AA)

        
        '''img_show_lado_a_lado([self.img,
                              cv2.cvtColor(self.bin, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(self.closed, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(self.norm, cv2.COLOR_GRAY2BGR),
                              self.img,
                              self.sure_fg_uint8,
                              self.sure_bg_uint8,
                              self.unknown,
                              self.img,
                              markers,
                              self.inundacao,
                              status_img
                              ], titles=['Original',
                                         'Binarizada',
                                         'Fechada',
                                         'Objeto',
                                         'Original',
                                         'Objeto',
                                         'Fundo',
                                         'Desconhecido',
                                         'Original',
                                         'Marcadores',
                                         'Inundação',
                                         'Resultado'], map='jet', save=save) '''
        return cupom_encontrado, poly_points

# %%
def main(start=1, end=None, input_dir="input"):
    if not os.path.isdir(input_dir):
        print("Diretório de entrada não encontrado:", input_dir)
        return

    image_names = sorted(
        file_name for file_name in os.listdir(input_dir)
        if file_name.lower().endswith(".jpg")
    )

    if not image_names:
        print("Nenhum arquivo JPG encontrado em:", input_dir)
        return

    start_index = max(start - 1, 0)
    selected_names = image_names[start_index:end]

    VP = FP = VN = FN = 0  # Inicializar contadores
    for image_name in selected_names:
        globals()["fname"] = image_name
        path = os.path.join(input_dir, image_name)
        print("Processando arquivo:", path)
        if os.path.isfile(path):
            filter = FilterWaterShed()
            markers, inundacao = filter.apply(path, LogBody("Message Inicial"))
            # Verificar se tem cupom (inundacao sempre 2d)
            # Veja se o rótulo maior é mais que 30% da inundação: positivo

            accepted, _ = filter.accepted(markers, inundacao)
                    
            # Contar VP ou FP
            if accepted:
                # positivo detectado. 
                if fname.endswith('P.jpg'):
                    print(f"VP (True Positive): {fname}")
                    VP += 1
                elif fname.endswith('N.jpg'):
                    print(f"FP (False Positive): {fname}")
                    FP += 1 
            else:
                # negativo detectado.
                if fname.endswith('P.jpg'):
                    print(f"FN (False Negative): {fname}")
                    FN += 1
                elif fname.endswith('N.jpg'):
                    print(f"VN (True Negative): {fname}")
                    VN += 1
        else:
            print("Arquivo não encontrado:", path)

    print(f"Resultados: VP={VP}, FP={FP}, VN={VN}, FN={FN}")
    print(f"Accuracy: {(VP + VN) / (VP + FP + VN + FN) * 100:.2f}%")
    print(f"Precision: {(VP) / (VP + FP) * 100:.2f}%" if (VP + FP) > 0 else "Precision: N/A")
    print(f"Recall: {(VP) / (VP + FN) * 100:.2f}%" if (VP + FN) > 0 else "Recall: N/A")



if __name__ == "__main__":

    # FIRST e LAST agora recortam a lista ordenada de JPGs encontrados
    FIRST = 0
    LAST = 4000
    INPUT_DIR = input
    main(FIRST, LAST, INPUT_DIR)




