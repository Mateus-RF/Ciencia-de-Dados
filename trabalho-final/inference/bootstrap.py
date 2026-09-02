"""
Estimação de parâmetros populacionais via reamostragem Bootstrap.

Implementa:
- Estatísticas amostrais observadas (média e desvio padrão).
- Reamostragem Bootstrap não-paramétrica (com reposição).
- Intervalo de Confiança (IC) 95% via percentis da distribuição empírica.
- Intervalo de Confiança (IC) 95% paramétrico (aproximação Normal / TCL).
- Verificação informal das condições do Teorema Central do Limite (TCL).
"""

import os
import logging

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bootstrap.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

Z_95 = 1.96


def sample_statistics(serie: pd.Series) -> dict:
    """Calcula média amostral (X̄), desvio padrão amostral (s) e N."""

    serie = serie.dropna()

    return {
        "n": int(serie.shape[0]),
        "media": float(serie.mean()),
        "desvio_padrao": float(serie.std(ddof=1)),
    }


def bootstrap_resample(serie: pd.Series, n_boot: int = 2000, random_state: int = 42) -> np.ndarray:
    """
    Gera a distribuição empírica Bootstrap da média a partir de `n_boot`
    réplicas com reposição do mesmo tamanho da amostra original.
    """

    serie = serie.dropna().to_numpy()
    n = serie.shape[0]

    rng = np.random.default_rng(random_state)

    medias_bootstrap = np.empty(n_boot)

    for i in range(n_boot):
        reamostra = rng.choice(serie, size=n, replace=True)
        medias_bootstrap[i] = reamostra.mean()

    logging.info(f"Bootstrap concluído com {n_boot} réplicas (n_amostra={n})")

    return medias_bootstrap


def bootstrap_ci(medias_bootstrap: np.ndarray, confianca: float = 95.0) -> tuple:
    """IC não-paramétrico: percentis 2.5% e 97.5% da distribuição empírica."""

    alfa = 100 - confianca
    inferior = np.percentile(medias_bootstrap, alfa / 2)
    superior = np.percentile(medias_bootstrap, 100 - alfa / 2)

    return float(inferior), float(superior)


def parametric_ci(media: float, desvio_padrao: float, n: int, z: float = Z_95) -> tuple:
    """IC paramétrico tradicional: X̄ ± z * (s / √N)."""

    erro_padrao = desvio_padrao / np.sqrt(n)
    margem = z * erro_padrao

    return float(media - margem), float(media + margem)


def check_clt_conditions(serie: pd.Series) -> dict:
    """
    Avalia informalmente se as condições do TCL parecem razoáveis para a
    variável em questão, com base no tamanho amostral (regra prática N >= 30)
    e na assimetria (skewness) da distribuição original.
    """

    serie = serie.dropna()
    n = serie.shape[0]
    assimetria = float(stats.skew(serie))

    n_suficiente = n >= 30
    baixa_assimetria = abs(assimetria) < 1.0

    if n_suficiente and baixa_assimetria:
        parecer = (
            "As condições do TCL parecem satisfeitas: o tamanho amostral é "
            "grande (N >= 30) e a assimetria da variável original é moderada, "
            "de modo que a distribuição amostral da média tende à Normalidade."
        )
    elif n_suficiente and not baixa_assimetria:
        parecer = (
            "O tamanho amostral é grande (N >= 30), o que favorece a "
            "aproximação Normal pelo TCL mesmo com a assimetria elevada "
            "observada na variável original."
        )
    else:
        parecer = (
            "O tamanho amostral é pequeno (N < 30) e/ou a assimetria da "
            "variável é elevada, portanto a aproximação Normal do TCL deve "
            "ser interpretada com cautela; o método Bootstrap é preferível "
            "por não depender dessa suposição."
        )

    return {
        "n": n,
        "assimetria": assimetria,
        "n_suficiente": n_suficiente,
        "baixa_assimetria": baixa_assimetria,
        "parecer": parecer,
    }


def run_bootstrap_analysis(
    df: pd.DataFrame,
    coluna: str,
    n_boot: int = 2000,
    confianca: float = 95.0,
    random_state: int = 42,
) -> dict:
    """
    Orquestra a análise completa de Bootstrap/IC para uma coluna numérica:
    estatísticas observadas, reamostragem, os dois métodos de IC e o
    diagnóstico das condições do TCL.
    """

    logging.info(f"Iniciando análise de bootstrap para a coluna '{coluna}'")

    serie = df[coluna]

    estatisticas = sample_statistics(serie)

    medias_bootstrap = bootstrap_resample(
        serie, n_boot=n_boot, random_state=random_state
    )

    ic_bootstrap = bootstrap_ci(medias_bootstrap, confianca=confianca)

    ic_parametrico = parametric_ci(
        estatisticas["media"], estatisticas["desvio_padrao"], estatisticas["n"]
    )

    diagnostico_tcl = check_clt_conditions(serie)

    resultado = {
        "coluna": coluna,
        "n_boot": n_boot,
        "confianca": confianca,
        "estatisticas": estatisticas,
        "medias_bootstrap": medias_bootstrap,
        "ic_bootstrap": ic_bootstrap,
        "ic_parametrico": ic_parametrico,
        "diagnostico_tcl": diagnostico_tcl,
    }

    logging.info(
        f"IC Bootstrap: {ic_bootstrap} | IC Paramétrico: {ic_parametrico}"
    )

    return resultado
