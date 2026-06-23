import pandas as pd
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
        "extractor.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def extract_data(path_csv):

    try:

        if not os.path.isfile(path_csv):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path_csv}"
            )
        
        logging.info(
            "Iniciando extração dos dados"
        )

        df = pd.read_csv(
            path_csv,
            encoding="utf-8"
        )

        linhas, colunas = df.shape

        logging.info(
            "Extração concluída"
        )

        logging.info(
            f"Dimensão inicial: {linhas} linhas x {colunas} colunas"
        )

        logging.info(
            f"Volumetria bruta: {linhas * colunas} valores"
        )

        return df

    except FileNotFoundError as erro:
        logging.error(
            erro
        )

        raise erro

    except Exception as erro:
        logging.error(
            f"Erro durante extração: {erro}"
        )

        raise erro