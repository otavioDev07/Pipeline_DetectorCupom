import sys
import os
import json
import cv2
import numpy as np
import re
import shutil

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_SHARPNESS = os.path.join(DIRETORIO_ATUAL, 'sharpness')
sys.path.append(PASTA_SHARPNESS)

try:
    from filter_sharpness import Filter_Sharpness
except ImportError as e:
    sys.stderr.write(f"\n[Erro Fatal] Nao consegui carregar o filtro na pasta 'sharpness': {e}\n")
    sys.exit(1)

def ordenar_pontos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    if len(sys.argv) < 4:
        sys.stderr.write("Uso: python3 recortar.py <imagem> <json_path> <output_dir>\n")
        sys.exit(1)

    img_path = sys.argv[1]
    json_path = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(img_path)
    matches = re.findall(r'\d+', filename)
    num_str = matches[0] if matches else "0000"

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        data = {"achou": False}

    achou = data.get("achou", False)
    pontos = data.get("pontos", [])

    if achou and len(pontos) == 4:
        img = cv2.imread(img_path)
        if img is None:
            sys.exit(1)

        pts = np.array(pontos, dtype="float32")
        rect = ordenar_pontos(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        filtro_nitidez = Filter_Sharpness()
        fft_score = filtro_nitidez.do(warped)
        
        if fft_score > 0.222:
            out_path = os.path.join(out_dir, f"{num_str}_P.jpg")
            cv2.imwrite(out_path, warped)
        else:
            sys.stderr.write(f"[Recorte] {num_str}: imagem rejeitada pelo borradez (FFT: {fft_score:.2f} < 225).\n")
            
            out_path = os.path.join(out_dir, f"{num_str}_N.jpg")
            shutil.copy(img_path, out_path)
    else:
        out_path = os.path.join(out_dir, f"{num_str}_N.jpg")
        shutil.copy(img_path, out_path)

if __name__ == "__main__":
    main()