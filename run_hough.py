import sys
import os
import json
import cv2
import numpy as np

HOUGH_DIR = os.path.join(os.path.dirname(__file__), 'filter_has_document')
sys.path.append(HOUGH_DIR)

from filter_hasdocumente import HasDocument

def calcular_geometria(p1, p2, p3, p4, shape):
    h_img, w_img = shape[:2]
    area_img = float(w_img * h_img)
    
    pts = np.array([p1, p2, p3, p4], dtype=np.float32)
    area_poly = cv2.contourArea(pts)
    relative_area = area_poly / area_img if area_img > 0 else 0.0

    max_cosine = 0.0
    for i in range(4):
        v1 = pts[(i + 3) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 1e-5 and n2 > 1e-5:
            cos = abs(np.dot(v1, v2) / (n1 * n2))
            max_cosine = max(max_cosine, cos)

    return float(relative_area * (1.0 - max_cosine))

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Uso: python3 run_hough.py <caminho_imagem>\n")
        sys.exit(1)

    img_path = sys.argv[1]
    sys.stderr.write(f"[Python-Hough] Processando: {img_path}\n")

    img = cv2.imread(img_path)
    if img is None:
        sys.stderr.write("[Python-Hough] Erro ao carregar imagem.\n")
        print(json.dumps({"achou": False, "score": 0.0, "pontos": []}))
        sys.exit(1)

    filtro = HasDocument(img)
    flag, p1, p2, p3, p4 = filtro.do()

    if flag == 1:
        score = calcular_geometria(p1, p2, p3, p4, img.shape)
        resultado = {
            "achou": True,
            "score": round(score, 6),
            "pontos": [
                [int(p1[0]), int(p1[1])],
                [int(p2[0]), int(p2[1])],
                [int(p3[0]), int(p3[1])],
                [int(p4[0]), int(p4[1])]
            ]
        }
    else:
        resultado = {"achou": False, "score": 0.0, "pontos": []}

    print(json.dumps(resultado, indent=2))

if __name__ == "__main__":
    main()