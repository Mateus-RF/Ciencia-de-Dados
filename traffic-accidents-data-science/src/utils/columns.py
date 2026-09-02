import pandas as pd


def resolve_column(df: pd.DataFrame, name: str) -> str:

    if name in df.columns:
        return name

    if name.upper() in df.columns:
        return name.upper()

    if name.lower() in df.columns:
        return name.lower()

    alvo = name.strip().lower()

    for coluna in df.columns:
        if str(coluna).strip().lower() == alvo:
            return coluna

    raise ValueError(
        f"Coluna '{name}' não encontrada no dataset. "
        f"Colunas disponíveis: {list(df.columns)}. "
        "Ajuste o nome no bloco de configuração de src/main.py."
    )


def resolve_numeric_columns(df: pd.DataFrame, exclude: list | None = None) -> list:

    exclude = set(exclude or [])

    return [
        coluna
        for coluna in df.select_dtypes(include="number").columns
        if coluna not in exclude
    ]
