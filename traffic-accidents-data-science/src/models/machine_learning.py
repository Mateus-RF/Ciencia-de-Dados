"""
Modelagem Preditiva Supervisionada: Classificação Binária.

Tarefa: prever se um acidente é GRAVE (MORTOS > 0 -> 1) ou NÃO GRAVE
(MORTOS == 0 -> 0), a partir de variáveis numéricas descritivas da
ocorrência (não relacionadas diretamente à contagem de mortos).

Compara dois algoritmos:
- Regressão Logística
- K-Vizinhos Mais Próximos (KNN)

Ambos dentro de um Pipeline (padronização + classificador), otimizados
via GridSearchCV com Validação Cruzada.
"""

import os
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "machine_learning.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def build_target(df: pd.DataFrame, severity_col: str) -> pd.Series:
    """Cria o alvo binário: 1 = acidente grave/fatal, 0 = caso contrário."""

    return (df[severity_col] > 0).astype(int)


def _evaluate(modelo, X_teste, y_teste) -> dict:

    y_pred = modelo.predict(X_teste)

    return {
        "matriz_confusao": confusion_matrix(y_teste, y_pred),
        "acuracia": float(accuracy_score(y_teste, y_pred)),
        "precisao": float(precision_score(y_teste, y_pred, zero_division=0)),
        "sensibilidade": float(recall_score(y_teste, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_teste, y_pred, zero_division=0)),
    }


def run_classification(
    df: pd.DataFrame,
    feature_cols: list,
    severity_col: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Orquestra o pipeline de classificação: separa features/target,
    treina Regressão Logística e KNN (cada um com GridSearchCV sobre
    seus hiperparâmetros) e retorna as métricas de ambos no teste.
    """

    logging.info(
        f"Iniciando classificação binária | features={feature_cols} "
        f"| alvo derivado de '{severity_col}'"
    )

    dados = df[feature_cols + [severity_col]].dropna()

    X = dados[feature_cols].to_numpy()
    y = build_target(dados, severity_col).to_numpy()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    resultados = {}

    # --- Regressão Logística ---
    pipeline_log = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])

    grade_log = {"clf__C": [0.01, 0.1, 1, 10, 100]}

    busca_log = GridSearchCV(
        pipeline_log, grade_log, cv=cv_folds, scoring="f1", n_jobs=-1
    )
    busca_log.fit(X_treino, y_treino)

    resultados["regressao_logistica"] = {
        "melhores_parametros": busca_log.best_params_,
        "melhor_f1_cv": float(busca_log.best_score_),
        **_evaluate(busca_log.best_estimator_, X_teste, y_teste),
    }

    logging.info(
        f"Regressão Logística | melhores parâmetros: {busca_log.best_params_}"
    )

    # --- KNN ---
    pipeline_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])

    grade_knn = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
    }

    busca_knn = GridSearchCV(
        pipeline_knn, grade_knn, cv=cv_folds, scoring="f1", n_jobs=-1
    )
    busca_knn.fit(X_treino, y_treino)

    resultados["knn"] = {
        "melhores_parametros": busca_knn.best_params_,
        "melhor_f1_cv": float(busca_knn.best_score_),
        **_evaluate(busca_knn.best_estimator_, X_teste, y_teste),
    }

    logging.info(
        f"KNN | melhores parâmetros: {busca_knn.best_params_}"
    )

    resultados["feature_cols"] = feature_cols
    resultados["n_treino"] = len(X_treino)
    resultados["n_teste"] = len(X_teste)

    return resultados
