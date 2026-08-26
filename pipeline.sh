#!/bin/bash

# --- 1. HIPERPARÂMETROS DE CORTE (EARLY EXITS) ---
THRESH_RDP=0.25
THRESH_HOUGH=0.22

if [ "$#" -ne 1 ]; then
    echo "Uso: ./pipeline.sh <caminho_da_imagem.jpg>"
    exit 1
fi

IMAGE="$1"
if [ ! -f "$IMAGE" ]; then
    echo "[Erro] Arquivo de imagem nao encontrado: $IMAGE"
    exit 1
fi

echo "==================================================="
echo "[Pipeline] Iniciando processamento: $IMAGE"
echo "==================================================="

# Ativa a bolha do Python silenciosamente
source .venv/bin/activate

# Cria uma pasta temporária para armazenar os JSONs intermediários
TMP_DIR="./tmp_pipeline"
mkdir -p "$TMP_DIR"

RDP_JSON="$TMP_DIR/rdp.json"
HOUGH_JSON="$TMP_DIR/hough.json"
WS_JSON="$TMP_DIR/ws.json"

echo "[1/4] Rodando RDP (C++)..."
./RDP+HoughProb/detector "$IMAGE" > "$RDP_JSON"

SCORE_RDP=$(python3 -c "import sys, json; print(json.load(sys.stdin).get('score', 0.0))" < "$RDP_JSON")

PASSED_RDP=$(python3 -c "print(1 if $SCORE_RDP >= $THRESH_RDP else 0)")
if [ "$PASSED_RDP" -eq 1 ]; then
    echo "[Early Exit 1] RDP atingiu confianca altissima ($SCORE_RDP). Finalizando!"
    FINAL_JSON="$RDP_JSON"
    
else
    echo "[2/4] RDP insuficiente ($SCORE_RDP). Rodando Hough (Python)..."
    python3 run_hough.py "$IMAGE" > "$HOUGH_JSON"
    
    SCORE_HOUGH=$(python3 -c "import sys, json; print(json.load(sys.stdin).get('score', 0.0))" < "$HOUGH_JSON")
    
    PASSED_HOUGH=$(python3 -c "print(1 if $SCORE_HOUGH >= $THRESH_HOUGH else 0)")
    if [ "$PASSED_HOUGH" -eq 1 ]; then
        echo "[Early Exit 2] Hough atingiu confianca intermediaria ($SCORE_HOUGH). Finalizando!"
        FINAL_JSON="$HOUGH_JSON"
        
    else
        echo "[3/4] Hough insuficiente ($SCORE_HOUGH). Acionando Watershed (Pesado)..."
        python3 run_watershed.py "$IMAGE" > "$WS_JSON"
        
        echo "[4/4] Sem Early Exits. Invocando Arbitro de Consenso (IoU)..."
        FINAL_JSON="$TMP_DIR/decisao_final.json"
        python3 modulo_decisao.py "$RDP_JSON" "$HOUGH_JSON" "$WS_JSON" > "$FINAL_JSON"
    fi
fi

echo "==================================================="
echo "RESULTADO FINAL:"
cat "$FINAL_JSON"
echo "==================================================="

OUTPUT_DIR="../result_pipeline"

echo "[5/5] Executando recorte e correcao de perspectiva..."
python3 recortar.py "$IMAGE" "$FINAL_JSON" "$OUTPUT_DIR"

rm -rf "$TMP_DIR"