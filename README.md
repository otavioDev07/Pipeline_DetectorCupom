# Pipeline: Detector de Cupom Fiscal

### 1. Dependências do Sistema (OpenCV C++)
O orquestrador exige as bibliotecas de desenvolvimento do OpenCV instaladas a nível de sistema operacional para compilar o detector RDP.

**Ubuntu / Debian (Linux):**
```bash
sudo apt-get update
sudo apt-get install libopencv-dev pkg-config
```

**macOS (Homebrew):**
```bash
brew install opencv pkg-config
```

### 2. Ambiente Python e Dependências
O script `pipeline.sh` ativa automaticamente um ambiente virtual chamado `.venv`. Para criá-lo e instalar as bibliotecas necessárias, execute os comandos abaixo na raiz do repositório:

```bash
# Cria o ambiente virtual local
python3 -m venv .venv

# Ativa o ambiente
source .venv/bin/activate

# Instala as dependências de Visão Computacional
pip install opencv-python numpy
```

### 3. Permissões de Execução
Certifique-se de que o script orquestrador tem permissão de execução no seu sistema:

```bash
chmod +x pipeline.sh
```

---

## Como Executar

O projeto está dividido em branches de acordo com o caso de uso (processamento unitário ou em lote). 

### Branch: main (Processamento Unitário)
Voltada para depuração e testes rápidos de extração em uma única imagem.

```bash
git checkout main
./pipeline.sh <caminho_da_imagem.jpg>
```
*A imagem processada e o recorte final serão gerados conforme o fluxo de log do terminal.*

### Branch: teste (Processamento em Lote / Dataset)
Voltada para validação de métricas contra datasets grandes. O script processa todas as imagens `.jpg` de um diretório de entrada e exporta os resultados classificados (`_P.jpg` para sucesso, `_N.jpg` para falha/rejeição) para um diretório de saída.

```bash
git checkout teste
./pipeline.sh <diretorio_de_entrada> <diretorio_de_saida>
```
*Exemplo:* `./pipeline.sh ./dataset_input ../dataset_output`

---
