"""
Teste de Hipóteses via Teste de Permutação (embaralhamento de rótulos).

Cenário adotado (domínio de Logística/Segurança Viária):

    Grupo A -> Acidentes GRAVES/FATAIS   (MORTOS > 0)
    Grupo B -> Acidentes LEVES/SEM VÍTIMAS (MORTOS == 0)

Métrica comparada: número de FERIDOS por ocorrência.
(O dataset não possui velocidade ou distância de frenagem registradas;
FERIDOS foi escolhida por ser a métrica contínua/discreta disponível mais
diretamente associada à gravidade humana do acidente, sem ser a própria
variável usada para definir os grupos.)

H0: não há diferença entre as médias de FERIDOS dos grupos A e B
    (mu_A - mu_B = 0).
H1: existe diferença entre as médias de FERIDOS dos grupos A e B
    (mu_A - mu_B != 0), teste bicaudal.
"""

import os
import logging

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "ab_testing.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

ALPHA = 0.05


def split_groups(df: pd.DataFrame, group_col: str, metric_col: str) -> tuple:
    """
    Divide o dataset em Grupo A (graves/fatais, group_col > 0) e
    Grupo B (leves/sem vítimas, group_col == 0), retornando os arrays
    da métrica de interesse para cada grupo.
    """

    grupo_a = df.loc[df[group_col] > 0, metric_col].dropna().to_numpy()
    grupo_b = df.loc[df[group_col] == 0, metric_col].dropna().to_numpy()

    logging.info(
        f"Grupo A (graves/fatais): n={len(grupo_a)} | "
        f"Grupo B (leves/sem vítimas): n={len(grupo_b)}"
    )

    return grupo_a, grupo_b


def observed_statistic(grupo_a: np.ndarray, grupo_b: np.ndarray) -> float:
    """Estatística de teste observada: diferença entre as médias amostrais."""

    return float(grupo_a.mean() - grupo_b.mean())


def permutation_test(
    grupo_a: np.ndarray,
    grupo_b: np.ndarray,
    n_perm: int = 2000,
    random_state: int = 42,
) -> dict:
    """
    Executa o Teste de Permutação: embaralha os rótulos dos grupos A/B
    repetidamente, recalculando a diferença de médias sob H0, e usa a
    proporção de diferenças tão ou mais extremas que a observada para
    estimar o p-valor empírico bicaudal.
    """

    diferenca_observada = observed_statistic(grupo_a, grupo_b)

    dados_combinados = np.concatenate([grupo_a, grupo_b])
    n_a = len(grupo_a)

    rng = np.random.default_rng(random_state)

    diferencas_permutadas = np.empty(n_perm)

    for i in range(n_perm):
        embaralhado = rng.permutation(dados_combinados)
        perm_a = embaralhado[:n_a]
        perm_b = embaralhado[n_a:]
        diferencas_permutadas[i] = perm_a.mean() - perm_b.mean()

    p_valor = float(
        np.mean(np.abs(diferencas_permutadas) >= np.abs(diferenca_observada))
    )

    logging.info(
        f"Diferença observada: {diferenca_observada:.4f} | p-valor: {p_valor:.4f}"
    )

    return {
        "diferenca_observada": diferenca_observada,
        "diferencas_permutadas": diferencas_permutadas,
        "p_valor": p_valor,
        "n_perm": n_perm,
    }


def run_ab_test(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    n_perm: int = 2000,
    alpha: float = ALPHA,
    random_state: int = 42,
) -> dict:
    """
    Orquestra o teste A/B completo: define os grupos, calcula a estatística
    observada, executa o teste de permutação e conclui sobre H0 ao nível
    de significância `alpha`.
    """

    logging.info("Iniciando Teste A/B (Teste de Permutação)")

    grupo_a, grupo_b = split_groups(df, group_col, metric_col)

    resultado = permutation_test(
        grupo_a, grupo_b, n_perm=n_perm, random_state=random_state
    )

    rejeita_h0 = resultado["p_valor"] < alpha

    conclusao = (
        f"Como o p-valor ({resultado['p_valor']:.4f}) é "
        f"{'menor' if rejeita_h0 else 'maior ou igual'} que alpha={alpha}, "
        f"{'rejeitamos' if rejeita_h0 else 'não rejeitamos'} a hipótese nula (H0)."
    )

    resultado.update({
        "group_col": group_col,
        "metric_col": metric_col,
        "alpha": alpha,
        "n_grupo_a": len(grupo_a),
        "n_grupo_b": len(grupo_b),
        "media_grupo_a": float(grupo_a.mean()),
        "media_grupo_b": float(grupo_b.mean()),
        "rejeita_h0": rejeita_h0,
        "conclusao": conclusao,
    })

    logging.info(conclusao)

    return resultado
