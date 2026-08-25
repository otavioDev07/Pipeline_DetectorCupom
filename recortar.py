import sys
import os
import json
import cv2
import numpy as np

def ordenar_pontos(pts):
    """
    Ordena os vértices na ordem geométrica esperada pelo warpPerspective:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
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

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        sys.stderr.write("[Recorte] Erro ao ler JSON final.\n")
        sys.exit(1)

    if not data.get("achou", False) or not data.get("pontos"):
        sys.stderr.write("[Recorte] Nenhum cupom valido no JSON. Abortando crop.\n")
        sys.exit(0)

    img = cv2.imread(img_path)
    if img is None:
        sys.stderr.write(f"[Recorte] Erro ao abrir a imagem original: {img_path}\n")
        sys.exit(1)

    pts = np.array(data["pontos"], dtype="float32")
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

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"crop_{filename}")
    
    cv2.imwrite(out_path, warped)
    print(f"[Recorte] Transformacao concluida! Salvo em: {out_path}")

if __name__ == "__main__":
    main()