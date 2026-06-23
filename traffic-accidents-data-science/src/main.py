import os

from extract.extractor import extract_data
from transform.cleaner import clean_data
from visualize import generate_visualizations


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

arquivo = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "datatran2007-2022.csv"
)

saida = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dados_limpos_final.csv"
)

def main():

    print("Extraindo dados...")

    df = extract_data(arquivo)

    print("Tratando dados...")

    df_limpo = clean_data(df)

    print("Salvando arquivo final...")

    df_limpo.to_csv(
        saida,
        index=False,
        encoding="utf-8"
    )

    print("Gerando visualizações...")

    generate_visualizations(df_limpo)

    print("Pipeline executado com sucesso!")


if __name__ == "__main__":

    main()