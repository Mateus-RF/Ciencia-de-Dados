import pandas as pd
import matplotlib.pyplot as plt
import os
import logging


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "figures"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

logging.basicConfig(
    filename=os.path.join(
        BASE_DIR,
        "logs",
        "visualize.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def accidents_by_year(df):

    if "ANO" not in df.columns:

        logging.warning(
            "Coluna ANO não encontrada"
        )

        return

    acidentes = (
        df["ANO"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10,5))
    plt.plot(
        acidentes.index,
        acidentes.values,
        marker="o"
    )

    plt.title("Evolução dos acidentes de trânsito por ano")
    plt.xlabel("Ano")
    plt.ylabel("Quantidade de acidentes")
    plt.ylim(bottom=0)
    plt.grid(True)

    caminho = os.path.join(
        REPORT_DIR,
        "acidentes_por_ano.png"
    )

    plt.savefig(
        caminho,
        bbox_inches="tight"
    )

    plt.close()

    logging.info("Gráfico temporal gerado")

def accidents_by_state(df):


    if "UF" not in df.columns:

        logging.warning(
            "Coluna UF não encontrada"
        )

        return
    estados = (
        df["UF"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10,6))
    plt.bar(
        estados.index,
        estados.values
    )

    plt.title("Estados com maior número de acidentes")
    plt.xlabel("Estado")
    plt.ylabel("Quantidade de acidentes")
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)

    caminho = os.path.join(
        REPORT_DIR,
        "acidentes_por_estado.png"
    )

    plt.savefig(
        caminho,
        bbox_inches="tight"
    )

    plt.close()

    logging.info("Gráfico por estado gerado")


def numerical_relationship(df):

    numericas = df.select_dtypes(include="number")

    if len(numericas.columns) < 2:

        logging.warning("Poucas variáveis numéricas")

        return

    coluna_x = numericas.columns[0]
    coluna_y = numericas.columns[1]

    plt.figure(figsize=(8,5))
    plt.scatter(
        df[coluna_x],
        df[coluna_y],
        alpha=0.5
    )

    plt.title(f"Relação entre {coluna_x} e {coluna_y}")
    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.grid(True)

    caminho = os.path.join(
        REPORT_DIR,
        "relacao_numerica.png"
    )

    plt.savefig(
        caminho,
        bbox_inches="tight"
    )

    plt.close()

    logging.info("Gráfico de dispersão gerado")


def generate_visualizations(df):

    logging.info("Iniciando camada visual")

    accidents_by_year(df)

    accidents_by_state(df)

    numerical_relationship(df)

    logging.info("Visualização finalizada")