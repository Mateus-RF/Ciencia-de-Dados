"""
Aprendizado Não Supervisionado: Redução de Dimensionalidade (PCA/SVD)
e Agrupamento (K-Means).
"""

import os
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "unsupervised.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def prepare_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Seleciona e padroniza (StandardScaler) as variáveis numéricas."""

    dados = df[feature_cols].dropna()

    scaler = StandardScaler()
    X_padronizado = scaler.fit_transform(dados)

    return X_padronizado


def run_pca(X_padronizado: np.ndarray, n_components: int = 2, random_state: int = 42) -> dict:
    """
    Aplica PCA (fundamentado em SVD) projetando os dados nos dois
    primeiros componentes principais.
    """

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_padronizado)

    variancia_explicada = pca.explained_variance_ratio_
    variancia_acumulada = float(np.sum(variancia_explicada))

    logging.info(
        f"PCA | variância explicada por componente: {variancia_explicada} "
        f"| acumulada: {variancia_acumulada:.4f}"
    )

    return {
        "X_pca": X_pca,
        "variancia_explicada": variancia_explicada,
        "variancia_acumulada": variancia_acumulada,
    }


def compute_elbow(X_padronizado: np.ndarray, k_min: int = 1, k_max: int = 10, random_state: int = 42) -> dict:
    """Calcula a inércia do K-Means para k no intervalo [k_min, k_max]."""

    k_range = list(range(k_min, k_max + 1))
    inercias = []

    for k in k_range:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        modelo.fit(X_padronizado)
        inercias.append(float(modelo.inertia_))

    logging.info(f"Método do cotovelo | k={k_range} | inércias={inercias}")

    return {"k_range": k_range, "inercias": inercias}


def choose_k_by_elbow(k_range: list, inercias: list) -> int:
    """
    Heurística simples para o "cotovelo": escolhe o k que maximiza a
    distância perpendicular entre a curva de inércia e a reta que une o
    primeiro e o último ponto (método da maior curvatura).
    """

    pontos = np.array(list(zip(k_range, inercias)), dtype=float)

    inicio, fim = pontos[0], pontos[-1]
    linha = fim - inicio
    norma_linha = np.linalg.norm(linha)

    distancias = []
    for ponto in pontos:
        vetor = ponto - inicio
        projecao = np.dot(vetor, linha) / norma_linha
        ponto_na_linha = inicio + projecao * (linha / norma_linha)
        distancias.append(np.linalg.norm(ponto - ponto_na_linha))

    indice_cotovelo = int(np.argmax(distancias))

    return int(k_range[indice_cotovelo])


def run_kmeans(X_padronizado: np.ndarray, k: int, random_state: int = 42) -> dict:
    """Ajusta o K-Means final com o número de clusters `k` escolhido."""

    modelo = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = modelo.fit_predict(X_padronizado)

    logging.info(f"K-Means final ajustado com k={k}")

    return {"k": k, "labels": labels, "inertia": float(modelo.inertia_)}


def run_unsupervised_analysis(
    df: pd.DataFrame,
    feature_cols: list,
    k_min: int = 1,
    k_max: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Orquestra a análise não-supervisionada completa: padronização, PCA,
    curva do cotovelo, seleção automática de k e K-Means final.
    """

    logging.info(f"Iniciando análise não-supervisionada | features={feature_cols}")

    X_padronizado = prepare_features(df, feature_cols)

    resultado_pca = run_pca(X_padronizado, random_state=random_state)

    resultado_cotovelo = compute_elbow(
        X_padronizado, k_min=k_min, k_max=k_max, random_state=random_state
    )

    k_escolhido = choose_k_by_elbow(
        resultado_cotovelo["k_range"], resultado_cotovelo["inercias"]
    )

    resultado_kmeans = run_kmeans(X_padronizado, k=k_escolhido, random_state=random_state)

    return {
        "feature_cols": feature_cols,
        "pca": resultado_pca,
        "cotovelo": resultado_cotovelo,
        "k_escolhido": k_escolhido,
        "kmeans": resultado_kmeans,
    }
