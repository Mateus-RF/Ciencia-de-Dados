import pandas as pd
import numpy as np
import logging
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)
os.makedirs(
    LOG_DIR,
    exist_ok=True
)

logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "cleaner.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def fix_encoding(df):

    colunas_texto = df.select_dtypes(
        include="object"
    ).columns

    for coluna in colunas_texto:

        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.encode(
                "latin1",
                errors="ignore"
            )
            .str.decode(
                "utf-8",
                errors="ignore"
            )
        )

    return df


def normalize_strings(df):

    colunas_texto = df.select_dtypes(
        include="object"
    ).columns

    for coluna in colunas_texto:

        df[coluna] = (
            df[coluna]
            .str.strip()
            .str.upper()
        )

    return df


def treat_missing(df):

    colunas_numericas = df.select_dtypes(
        include=np.number
    ).columns

    for coluna in colunas_numericas:

        df[coluna] = df[coluna].fillna(
            df[coluna].median()
        )

    df = df.dropna(
        how="all"
    )

    return df


def remove_outliers_iqr(df):

    colunas = df.select_dtypes(
        include=np.number
    ).columns

    for coluna in colunas:
        Q1 = df[coluna].quantile(
            0.25
        )

        Q3 = df[coluna].quantile(
            0.75
        )

        IQR = Q3 - Q1

        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR

        df = df[
            (df[coluna] >= limite_inf)
            &
            (df[coluna] <= limite_sup)
        ]

    return df

def clean_data(df):

    logging.info("Iniciando limpeza")

    df = fix_encoding(df)
    df = normalize_strings(df)
    df = treat_missing(df)
    df = remove_outliers_iqr(df)

    logging.info(f"Dataset final: {df.shape}")

    return df