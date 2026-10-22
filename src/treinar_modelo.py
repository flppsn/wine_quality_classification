"""
Pipeline do modelo final: carrega a base, prepara os dados, treina a
Regressão Logística e salva as métricas em results/.

Uso (a partir da pasta raiz do projeto):
    python src/treinar_modelo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_DADOS = RAIZ / "data" / "Base.csv"
PASTA_RESULTADOS = RAIZ / "results"

FEATURES_ORIGINAIS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
COLUNAS_LOG = ["chlorides", "residual sugar", "sulphates",
               "total sulfur dioxide", "free sulfur dioxide"]
FEATURES_NOVAS = ["sulfatos_por_cloretos", "razao_so2_livre"]
FEATURES_MODELO = FEATURES_ORIGINAIS + FEATURES_NOVAS

CORTE_FINAL = 0.45
RANDOM_STATE = 42


def carregar_dados(caminho=ARQUIVO_DADOS):
    """Lê a base e cria a variável alvo (1 = nota >= 7)."""
    df = pd.read_csv(caminho)
    df["qualidade_alta"] = (df["quality"] >= 7).astype(int)
    return df


def preparar_dados(df):
    """Remove duplicatas, aplica log nas variáveis assimétricas e cria as
    variáveis novas."""
    limpo = (df.drop_duplicates(subset=FEATURES_ORIGINAIS + ["quality"])
               .reset_index(drop=True))
    proc = limpo.copy()
    proc[COLUNAS_LOG] = np.log(proc[COLUNAS_LOG])
    proc["sulfatos_por_cloretos"] = np.log(limpo["sulphates"] / limpo["chlorides"])
    proc["razao_so2_livre"] = (limpo["free sulfur dioxide"]
                               / limpo["total sulfur dioxide"])
    return proc


def dividir_dados(proc):
    """Divide em treino (80%) e teste (20%), estratificado pelo alvo."""
    X = proc[FEATURES_MODELO]
    y = proc["qualidade_alta"]
    return train_test_split(X, y, test_size=0.2, stratify=y,
                            random_state=RANDOM_STATE)


def treinar_modelo(X_train, y_train):
    """Padroniza com base no treino e treina a Regressão Logística."""
    scaler = StandardScaler().fit(X_train)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(scaler.transform(X_train), y_train)
    return scaler, modelo


def avaliar_modelo(scaler, modelo, X_test, y_test, corte=CORTE_FINAL):
    """Calcula as métricas do modelo no conjunto de teste."""
    prob = modelo.predict_proba(scaler.transform(X_test))[:, 1]
    pred = (prob >= corte).astype(int)
    return {
        "precisão": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f0.5": fbeta_score(y_test, pred, beta=0.5),
        "auc": roc_auc_score(y_test, prob),
        "vinhos indicados como alta qualidade": int(pred.sum()),
        "acertos": int(((pred == 1) & (y_test == 1)).sum()),
    }


def main():
    df = carregar_dados()
    proc = preparar_dados(df)
    X_train, X_test, y_train, y_test = dividir_dados(proc)
    scaler, modelo = treinar_modelo(X_train, y_train)
    metricas = avaliar_modelo(scaler, modelo, X_test, y_test)

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    saida = PASTA_RESULTADOS / "metricas_modelo_final.csv"
    pd.Series({k: round(v, 3) for k, v in metricas.items()},
              name="valor", dtype=object).to_csv(saida, header=True)

    print(f"Modelo final: Regressão Logística (corte {CORTE_FINAL})")
    for nome, valor in metricas.items():
        print(f"  {nome}: {round(valor, 3)}")
    print(f"Métricas salvas em: {saida}")


if __name__ == "__main__":
    main()
