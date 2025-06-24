import json
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ========== CONFIGURATION ==========
ARQUIVOS_ORIGINAIS = [
    Path("./dataset/dev.json"),
    Path("./dataset/train.json")
]
ARQUIVO_CLASSIFICADO = Path(sys.argv[1])
# ===================================

CATEGORIAS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking"
]

def carregar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def comparar_labels(originais, classificados_dict):
    y_true = []
    y_pred = []

    for item in originais:
        claim = item.get("claim")
        label_real = item.get("label")  # já está em inglês
        label_predita = classificados_dict.get(claim)  # também em inglês

        if not (claim and label_real and label_predita):
            continue

        y_true.append(label_real)
        y_pred.append(label_predita)

    return y_true, y_pred

def imprimir_matriz_confusao(y_true, y_pred, categorias):
    matriz = confusion_matrix(y_true, y_pred, labels=categorias)

    print("📊 Confusion Matrix (True Label × Predicted Label):\n")
    cabecalho = f"{'':35}" + "".join(f"{c[:20]:>25}" for c in categorias)
    print(cabecalho)
    for i, linha in enumerate(matriz):
        nome = categorias[i]
        linha_str = f"{nome[:33]:35}" + "".join(f"{n:>25}" for n in linha)
        print(linha_str)

def imprimir_relatorio(y_true, y_pred):
    print(f"\n📊 Overall Evaluation")
    print(f"Claims compared: {len(y_true)}")
    acertos = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    print(f"Exact matches: {acertos} ({acertos / len(y_true):.1%})\n")

    imprimir_matriz_confusao(y_true, y_pred, CATEGORIAS)

    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=CATEGORIAS, digits=3, zero_division=0))

if __name__ == "__main__":
    # Load model predictions
    classificados = carregar_json(ARQUIVO_CLASSIFICADO)
    classificados_dict = {
        item["claim"]: item["label"]
        for item in classificados
        if "claim" in item and "label" in item
    }

    y_true_total = []
    y_pred_total = []

    for arquivo in ARQUIVOS_ORIGINAIS:
        print(f"🔍 Processing {arquivo.name}")
        originais = carregar_json(arquivo)
        y_true, y_pred = comparar_labels(originais, classificados_dict)
        y_true_total.extend(y_true)
        y_pred_total.extend(y_pred)

    imprimir_relatorio(y_true_total, y_pred_total)
