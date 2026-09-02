import sys
import json
import cv2
import numpy as np

def carregar_json(caminho):
    try:
        with open(caminho, 'r') as f:
            return json.load(f)
    except Exception:
        return {"achou": False, "score": 0.0, "pontos": []}

def ordenar_pontos(pts):
    """
    Ordena os 4 pontos de um quadrilátero de forma consistente:
    [Topo-Esquerdo, Topo-Direito, Base-Direito, Base-Esquerdo]
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # A soma das coordenadas (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Topo-Esquerdo tem a menor soma
    rect[2] = pts[np.argmax(s)] # Base-Direito tem a maior soma
    
    # A diferença das coordenadas (x - y) ou (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Topo-Direito tem a menor diferença
    rect[3] = pts[np.argmax(diff)] # Base-Esquerdo tem a maior diferença
    
    return rect

def calcular_iou(pts1, pts2):
    p1 = np.array(pts1, dtype=np.float32)
    p2 = np.array(pts2, dtype=np.float32)
    
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    if area1 == 0 or area2 == 0:
        return 0.0
        
    ret, inter_poly = cv2.intersectConvexConvex(p1, p2)
    if ret <= 0.0:
        return 0.0
        
    area_inter = cv2.contourArea(inter_poly)
    area_union = area1 + area2 - area_inter
    
    if area_union <= 0:
        return 0.0
        
    return area_inter / area_union

def media_dos_pontos(pts1, pts2):
    """
    Ordena ambos os vetores de pontos e calcula o ponto médio perfeito
    entre as duas detecções para gerar um quadrilátero mesclado.
    """
    p1_ordenado = ordenar_pontos(np.array(pts1, dtype=np.float32))
    p2_ordenado = ordenar_pontos(np.array(pts2, dtype=np.float32))
    
    pontos_media = (p1_ordenado + p2_ordenado) / 2.0
    
    # Retorna como lista de inteiros no padrão do JSON original
    return np.int32(pontos_media).tolist()

def main():
    if len(sys.argv) < 4:
        sys.stderr.write("Uso: python3 modulo_decisao.py <json_rdp> <json_hough> <json_watershed>\n")
        sys.exit(1)

    rdp = carregar_json(sys.argv[1])
    hough = carregar_json(sys.argv[2])
    ws = carregar_json(sys.argv[3])

    # --- PARÂMETROS DO PAPER DESIGN ---
    LIMIAR_IOU = 0.80
    TRAVA_FALLBACK = 0.22

    candidatos = {"RDP": rdp, "Hough": hough, "Watershed": ws}
    validos = {nome: dados for nome, dados in candidatos.items() if dados.get("achou")}

    resultado_final = {"achou": False, "score": 0.0, "pontos": []}
    estrategia_escolhida = "REJEITADO (Nenhum candidato valido)"
    consenso_encontrado = False

    nomes_validos = list(validos.keys())
    
    # 1. Busca por Consenso (IoU >= 0.80 com fusão de pontos)
    for i in range(len(nomes_validos)):
        if consenso_encontrado:
            break
        for j in range(i + 1, len(nomes_validos)):
            n1, n2 = nomes_validos[i], nomes_validos[j]
            
            pontos_n1 = validos[n1]["pontos"]
            pontos_n2 = validos[n2]["pontos"]
            
            iou = calcular_iou(pontos_n1, pontos_n2)
            
            if iou >= LIMIAR_IOU:
                # O consenso foi atingido! Vamos fundir os pontos (Média)
                pontos_fundidos = media_dos_pontos(pontos_n1, pontos_n2)
                
                # Para a nota final, matematicamente pegamos a maior entre as duas, 
                # já que ambas descrevem a mesma área real
                score_final = max(validos[n1]["score"], validos[n2]["score"])
                
                resultado_final = {
                    "achou": True,
                    "score": round(score_final, 6),
                    "pontos": pontos_fundidos
                }
                estrategia_escolhida = f"CONSENSO BLEND ({n1} e {n2}) | IoU: {iou:.2f}"
                consenso_encontrado = True
                break

    # 2. Fallback Isolado (Se não bateu 80%)
    if not consenso_encontrado and len(validos) > 0:
        melhor_nome = max(validos, key=lambda k: validos[k]["score"])
        melhor_candidato = validos[melhor_nome]
        
        if melhor_candidato["score"] >= TRAVA_FALLBACK:
            resultado_final = melhor_candidato
            estrategia_escolhida = f"FALLBACK ({melhor_nome})"
        else:
            estrategia_escolhida = "REJEITADO (Falha no Fallback Minimo)"

    sys.stderr.write(f"[Modulo-Decisao] {estrategia_escolhida} | Score Final: {resultado_final['score']:.4f}\n")
    print(json.dumps(resultado_final, indent=2))

if __name__ == "__main__":
    main()