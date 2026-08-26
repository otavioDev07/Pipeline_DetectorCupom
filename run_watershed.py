import sys
import os
import json
import cv2
import numpy as np

INUNDACAO_DIR = os.path.join(os.path.dirname(__file__), 'Inundacao')
sys.path.append(INUNDACAO_DIR)

try:
    from inundacao import FilterWaterShed
    from src.log.log_body import LogBody 
except ImportError as e:
    sys.stderr.write(f"[Python-Watershed] Erro de importacao interna do colega: {e}\n")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Uso: python3 run_watershed.py <caminho_imagem>\n")
        sys.exit(1)

    img_path = sys.argv[1]  
    sys.stderr.write(f"[Python-Watershed] Processando: {img_path}\n")

    if not os.path.exists(img_path):
        print(json.dumps({"achou": False, "score": 0.0, "pontos": []}))
        sys.exit(1)

    try:
        filter_ws = FilterWaterShed()
        log_real = LogBody(message=json.dumps({"arquivo": img_path}))
        markers, inundacao_map = filter_ws.apply(img_path, log_real)
        cupom_encontrado, poly_points = filter_ws.accepted(markers, inundacao_map, save=False)

        if cupom_encontrado and poly_points is not None:
            escala_reversa = 1.0 / filter_ws.escala if hasattr(filter_ws, 'escala') else 2.0
            pts_orig = np.float32(poly_points * escala_reversa)

            rect = cv2.minAreaRect(pts_orig)
            box = cv2.boxPoints(rect)
            pts4 = np.int32(box)

            h_img, w_img = filter_ws.img.shape[:2]
            area_img = float(w_img * h_img * (escala_reversa ** 2))
            area_poly = cv2.contourArea(pts4)
            rel_area = area_poly / area_img if area_img > 0 else 0.0

            if rel_area > 0.80:
                sys.stderr.write(f"[Python-Watershed] Alerta: Vazamento de borda detectado (Area: {rel_area:.2f}). Rejeitando.\n")
                resultado = {"achou": False, "score": 0.0, "pontos": []}
            else:
                max_cos = 0.0
                for i in range(4):
                    v1 = pts4[(i + 3) % 4] - pts4[i]
                    v2 = pts4[(i + 1) % 4] - pts4[i]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-5 and n2 > 1e-5:
                        max_cos = max(max_cos, abs(np.dot(v1, v2) / (n1 * n2)))

                score = rel_area * (1.0 - max_cos)

                resultado = {
                    "achou": True,
                    "score": round(float(score), 6),
                    "pontos": pts4.tolist()
                }
        else:
            resultado = {"achou": False, "score": 0.0, "pontos": []}

    except Exception as e:
        sys.stderr.write(f"[Python-Watershed] Erro na execucao interna: {e}\n")
        resultado = {"achou": False, "score": 0.0, "pontos": []}

    # 6. Saída JSON
    print(json.dumps(resultado, indent=2))

if __name__ == "__main__":
    main()