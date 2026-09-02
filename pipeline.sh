#!/bin/bash

# HIPERPARÂMETROS DE CORTE (EARLY EXITS)
THRESH_RDP=0.30
THRESH_HOUGH=0.28

# --- 2. VALIDAÇÃO DE ENTRADA (AGORA PARA PASTAS) --
if [ "$#" -ne 2 ]; then
    echo "Uso: ./pipeline.sh <PASTA_DE_ENTRADA> <PASTA_DE_SAIDA>"
    echo "Exemplo: ./pipeline.sh ./input ../result_pipeline"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$INPUT_DIR" ]; then
    echo "[Erro] A pasta de entrada '$INPUT_DIR' nao existe!"
    exit 1
fi

echo "==================================================="
echo "[Pipeline] Iniciando processamento em lote da pasta: $INPUT_DIR"
echo "==================================================="

# --- 3. PREPARAÇÃO DO AMBIENTE ---
source .venv/bin/activate

mkdir -p "$OUTPUT_DIR"
REJEITADAS_DIR="$OUTPUT_DIR/REJEITADAS"
mkdir -p "$REJEITADAS_DIR"

TMP_DIR="./tmp_pipeline"
mkdir -p "$TMP_DIR"

RDP_JSON="$TMP_DIR/rdp.json"
HOUGH_JSON="$TMP_DIR/hough.json"
WS_JSON="$TMP_DIR/ws.json"
FINAL_JSON="$TMP_DIR/decisao_final.json"

# --- VERIFICAÇÃO E COMPILAÇÃO DO C++ ---
DETECTOR_BIN="./RDP+HoughProb/detector"
if [ ! -f "$DETECTOR_BIN" ]; then
    echo "🔨 Compilando motor C++..."
    g++ -std=c++17 RDP+HoughProb/detector.cpp -o "$DETECTOR_BIN" $(pkg-config --cflags --libs opencv4)
    if [ $? -ne 0 ]; then 
        echo "[Erro] Falha ao compilar o detector C++."
        exit 1 
    fi
fi

# Contagem considerando arquivos webp 
TOTAL=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.webp" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
COUNT=0
REJEITADAS=()

shopt -s nullglob
for IMAGE in "$INPUT_DIR"/*.{webp,WEBP,jpg,JPG,jpeg,JPEG,png,PNG}; do
    [ -f "$IMAGE" ] || continue 
    
    COUNT=$((COUNT+1))
    FILENAME=$(basename "$IMAGE")
    
    # 1. ACIONA O C++
    "$DETECTOR_BIN" "$IMAGE" > "$RDP_JSON" 2>/dev/null
    
    SCORE_RDP=$(python3 -c 'import sys, json
try: print(json.load(sys.stdin).get("score", 0.0))
except: print(0.0)' < "$RDP_JSON" 2>/dev/null)
    
    SCORE_RDP=${SCORE_RDP:-0.0}
    PASSED_RDP=$(python3 -c "print(1 if $SCORE_RDP >= $THRESH_RDP else 0)")
    
    if [ "$PASSED_RDP" -eq 1 ]; then
        cp "$RDP_JSON" "$FINAL_JSON"
        ESTRATEGIA="RDP"
    else
        # 2. ACIONA O HOUGH
        python3 run_hough.py "$IMAGE" > "$HOUGH_JSON" 2>/dev/null
        
        SCORE_HOUGH=$(python3 -c 'import sys, json
try: print(json.load(sys.stdin).get("score", 0.0))
except: print(0.0)' < "$HOUGH_JSON" 2>/dev/null)
        
        SCORE_HOUGH=${SCORE_HOUGH:-0.0}
        PASSED_HOUGH=$(python3 -c "print(1 if $SCORE_HOUGH >= $THRESH_HOUGH else 0)")
        
        if [ "$PASSED_HOUGH" -eq 1 ]; then
            cp "$HOUGH_JSON" "$FINAL_JSON"
            ESTRATEGIA="Hough"
        else
            # 3. ÚLTIMO RECURSO (Watershed direto, sem Fast Reject)
            python3 run_watershed.py "$IMAGE" > "$WS_JSON" 2>/dev/null
            python3 modulo_decisao.py "$RDP_JSON" "$HOUGH_JSON" "$WS_JSON" > "$FINAL_JSON" 2>/dev/null
            ESTRATEGIA="Consenso/Fallback"
        fi
    fi

    # Aplica o recorte
    python3 recortar.py "$IMAGE" "$FINAL_JSON" "$OUTPUT_DIR"
    STATUS_RECORTE=$?

    if [ $STATUS_RECORTE -eq 0 ]; then
        echo "[$COUNT/$TOTAL] $FILENAME -> Processada via $ESTRATEGIA"
    else
        echo "[$COUNT/$TOTAL] $FILENAME -> REJEITADA"
        REJEITADAS+=("$FILENAME")
        cp "$IMAGE" "$REJEITADAS_DIR/"
    fi
done
shopt -u nullglob

# --- 5. LIMPEZA ---
rm -rf "$TMP_DIR"
echo "==================================================="
echo "Processamento concluido! Imagens salvas em: $OUTPUT_DIR"
echo "Total de imagens rejeitadas: ${#REJEITADAS[@]}"
echo "Copia das rejeitadas salva em: $REJEITADAS_DIR"
echo "==================================================="

if [ ${#REJEITADAS[@]} -gt 0 ]; then
    TXT_REJEITADAS="$REJEITADAS_DIR/lista_rejeitadas.txt"
    > "$TXT_REJEITADAS" # Limpa o arquivo caso ele já exista de uma rodada anterior

    echo "LISTA DE IMAGENS REJEITADAS:"
    for IMG in "${REJEITADAS[@]}"; do
        echo " - $IMG"
        echo "$IMG" >> "$TXT_REJEITADAS"
    done
    echo "==================================================="
    echo "Arquivo TXT com os nomes gerado em: $TXT_REJEITADAS"
    echo "==================================================="
fi