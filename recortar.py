import sys
import os
import json
import cv2
import numpy as np

# Mapeamento do filtro de nitidez
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_SHARPNESS = os.path.join(DIRETORIO_ATUAL, 'sharpness')
sys.path.append(PASTA_SHARPNESS)

try:
    from filter_sharpness import Filter_Sharpness
except ImportError as e:
    sys.stderr.write(f"\n[Erro Fatal] Nao foi possivel carregar o filtro de nitidez: {e}\n")
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

    # Garante o mesmo nome base da imagem, convertendo a extensão de saída para .jpg
    filename = os.path.basename(img_path)
    base_name, _ = os.path.splitext(filename)
    out_filename = f"{base_name}.jpg"

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

        # Avaliação de Nitidez FFT
        filtro_nitidez = Filter_Sharpness()
        fft_score = filtro_nitidez.do(warped)
        
        if fft_score > 0.222:
            out_path = os.path.join(out_dir, out_filename)
            cv2.imwrite(out_path, warped)
            sys.exit(0) # Código 0 = Sucesso
        else:
            sys.stderr.write(f"[Recorte] {filename}: rejeitada por borradez (FFT: {fft_score:.3f}).\n")
            sys.exit(1) # Código 1 = Rejeitada por borradez
    else:
        sys.exit(1) # Código 1 = Rejeitada por falta de cupom

if __name__ == "__main__":
    main()