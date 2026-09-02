"""
Modelagem Preditiva Supervisionada: Regressão Linear Múltipla.

Variável resposta (Y): FERIDOS (quantidade de feridos por ocorrência).
Variáveis preditoras: KM (quilometragem da ocorrência) e MORTOS
(quantidade de vítimas fatais), sob a ótica do princípio Ceteris Paribus.
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "regression.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def fit_multiple_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Ajusta um modelo de Regressão Linear Múltipla (Y ~ X1 + X2 + ...),
    dividindo os dados em treino/teste e reportando os coeficientes
    estimados, o R² e o RMSE no conjunto de teste.
    """

    logging.info(
        f"Ajustando regressão: Y='{y_col}' ~ X={x_cols}"
    )

    dados = df[[y_col] + x_cols].dropna()

    X = dados[x_cols].to_numpy()
    y = dados[y_col].to_numpy()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)

    r2 = float(r2_score(y_teste, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_teste, y_pred)))

    coeficientes = {
        "intercepto_beta0": float(modelo.intercept_),
        **{
            f"beta_{coluna}": float(coef)
            for coluna, coef in zip(x_cols, modelo.coef_)
        },
    }

    logging.info(f"Coeficientes: {coeficientes} | R2={r2:.4f} | RMSE={rmse:.4f}")

    return {
        "y_col": y_col,
        "x_cols": x_cols,
        "coeficientes": coeficientes,
        "r2": r2,
        "rmse": rmse,
        "n_treino": len(X_treino),
        "n_teste": len(X_teste),
        "y_teste": y_teste,
        "y_pred": y_pred,
    }
