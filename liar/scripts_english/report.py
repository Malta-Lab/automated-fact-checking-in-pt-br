import json
import sys
from collections import Counter
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ========== CONFIGURAÇÕES ==========
ARQUIVOS_ORIGINAIS = [
    Path("./dataset/test.jsonl"),
    Path("./dataset/train.jsonl"),
    Path("./dataset/valid.jsonl"),
]
ARQUIVO_CLASSIFICADO = Path(sys.argv[1])
# ===================================

# Labels originais do dataset LIAR
CATEGORIAS = [
    "true",
    "false",
    "half-true",
    "pants-fire",
    "barely-true",
    "mostly-true"
]

def carregar_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]

def carregar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def comparar_labels(originais, classificados_dict):
    y_true = []
    y_pred = []

    for item in originais:
        statement = item.get("statement")
        label_real = item.get("label")
        label_predita = classificados_dict.get(statement)

        if not (statement and label_real and label_predita):
            continue

        # Verifica se o rótulo predito está nas categorias válidas
        label_predita = label_predita.lower().strip()
        if label_predita not in CATEGORIAS:
            continue

        y_true.append(label_real)
        y_pred.append(label_predita)

    return y_true, y_pred

def imprimir_matriz_confusao(y_true, y_pred, categorias):
    matriz = confusion_matrix(y_true, y_pred, labels=categorias)

    print("\n📊 Matriz de Confusão (label verdadeiro × classificação do modelo):\n")
    cabecalho = f"{'':35}" + "".join(f"{c[:20]:>25}" for c in categorias)
    print(cabecalho)
    for i, linha in enumerate(matriz):
        nome = categorias[i]
        linha_str = f"{nome[:33]:35}" + "".join(f"{n:>25}" for n in linha)
        print(linha_str)

def imprimir_relatorio(y_true, y_pred):
    print(f"\n📊 Avaliação geral")
    print(f"Claims comparadas: {len(y_true)}")
    acertos = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    print(f"Acertos exatos: {acertos} ({acertos / len(y_true):.1%})\n")

    imprimir_matriz_confusao(y_true, y_pred, CATEGORIAS)

    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=CATEGORIAS, digits=3, zero_division=0))

if __name__ == "__main__":
    # Carrega classificações feitas pelo modelo
    classificados = carregar_json(ARQUIVO_CLASSIFICADO)
    classificados_dict = {
        item["statement"]: item["label"]
        for item in classificados
        if "statement" in item and "label" in item
    }

    y_true_total = []
    y_pred_total = []

    for arquivo in ARQUIVOS_ORIGINAIS:
        print(f"🔍 Processando {arquivo.name}")
        originais = carregar_jsonl(arquivo)
        y_true, y_pred = comparar_labels(originais, classificados_dict)
        y_true_total.extend(y_true)
        y_pred_total.extend(y_pred)

    imprimir_relatorio(y_true_total, y_pred_total)
