import pandas as pd
import numpy as np
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


def plot_bootstrap_distribution(resultado_bootstrap):
   
    medias_bootstrap = resultado_bootstrap["medias_bootstrap"]
    ic_bootstrap = resultado_bootstrap["ic_bootstrap"]
    ic_parametrico = resultado_bootstrap["ic_parametrico"]
    media_observada = resultado_bootstrap["estatisticas"]["media"]
    coluna = resultado_bootstrap["coluna"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        medias_bootstrap,
        bins=40,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
    )

    plt.axvline(
        media_observada, color="black", linestyle="-", linewidth=2,
        label=f"Média observada = {media_observada:.2f}"
    )

    plt.axvline(
        ic_bootstrap[0], color="#DD8452", linestyle="--", linewidth=2,
        label=f"IC Bootstrap 95% [{ic_bootstrap[0]:.2f}, {ic_bootstrap[1]:.2f}]"
    )
    plt.axvline(ic_bootstrap[1], color="#DD8452", linestyle="--", linewidth=2)

    plt.axvline(
        ic_parametrico[0], color="#55A868", linestyle=":", linewidth=2,
        label=f"IC Paramétrico 95% [{ic_parametrico[0]:.2f}, {ic_parametrico[1]:.2f}]"
    )
    plt.axvline(ic_parametrico[1], color="#55A868", linestyle=":", linewidth=2)

    plt.title(f"Distribuição Bootstrap da Média de {coluna}")
    plt.xlabel(f"Média amostral de {coluna}")
    plt.ylabel("Frequência (réplicas Bootstrap)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(BASE_DIR, "distribuicao_bootstrap.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de distribuição bootstrap gerado")


def plot_permutation_distribution(resultado_ab):

    diferencas_permutadas = resultado_ab["diferencas_permutadas"]
    diferenca_observada = resultado_ab["diferenca_observada"]
    p_valor = resultado_ab["p_valor"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        diferencas_permutadas,
        bins=40,
        color="#8172B2",
        alpha=0.75,
        edgecolor="white",
    )

    plt.axvline(
        diferenca_observada, color="crimson", linestyle="-", linewidth=2,
        label=f"Diferença observada = {diferenca_observada:.3f} (p-valor = {p_valor:.4f})"
    )
    plt.axvline(-diferenca_observada, color="crimson", linestyle="--", linewidth=1.5)

    plt.title("Distribuição da Diferença de Médias sob H0 (Teste de Permutação)")
    plt.xlabel("Diferença de médias (Grupo A - Grupo B) sob rótulos embaralhados")
    plt.ylabel("Frequência (permutações)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(BASE_DIR, "distribuicao_permutacao.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de distribuição de permutação gerado")


def plot_elbow_curve(resultado_cotovelo, k_escolhido):

    k_range = resultado_cotovelo["k_range"]
    inercias = resultado_cotovelo["inercias"]

    plt.figure(figsize=(9, 6))

    plt.plot(k_range, inercias, marker="o", color="#4C72B0")

    plt.axvline(
        k_escolhido, color="crimson", linestyle="--",
        label=f"k escolhido = {k_escolhido}"
    )

    plt.title("Método do Cotovelo para Seleção de k (K-Means)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia (soma dos quadrados intra-cluster)")
    plt.xticks(k_range)
    plt.legend()
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(BASE_DIR, "curva_cotovelo_kmeans.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de curva do cotovelo gerado")


def plot_pca_projection(resultado_pca):

    X_pca = resultado_pca["X_pca"]
    variancia_explicada = resultado_pca["variancia_explicada"]

    plt.figure(figsize=(9, 7))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color="#4C72B0", s=15)

    plt.title("Projeção PCA das Observações (2 Componentes Principais)")
    plt.xlabel(f"PC1 ({variancia_explicada[0] * 100:.1f}% da variância)")
    plt.ylabel(f"PC2 ({variancia_explicada[1] * 100:.1f}% da variância)")
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(BASE_DIR, "pca_projecao.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de projeção PCA gerado")


def plot_kmeans_clusters(resultado_pca, resultado_kmeans):
    

    X_pca = resultado_pca["X_pca"]
    labels = resultado_kmeans["labels"]
    k = resultado_kmeans["k"]

    plt.figure(figsize=(9, 7))

    dispersao = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="tab10", alpha=0.6, s=15
    )

    plt.title(f"Clusters K-Means Projetados no Espaço PCA (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(
        handles=dispersao.legend_elements()[0],
        labels=[f"Cluster {i}" for i in range(k)],
        title="Clusters",
    )
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(BASE_DIR, "clusters_kmeans.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de clusters K-Means gerado")


def plot_confusion_matrix(matriz_confusao, nome_modelo):

    plt.figure(figsize=(5, 4.5))

    plt.imshow(matriz_confusao, cmap="Blues")
    plt.title(f"Matriz de Confusão — {nome_modelo}")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.xticks([0, 1], ["Não Grave", "Grave"])
    plt.yticks([0, 1], ["Não Grave", "Grave"])
    plt.colorbar()

    for i in range(matriz_confusao.shape[0]):
        for j in range(matriz_confusao.shape[1]):
            plt.text(
                j, i, str(matriz_confusao[i, j]),
                ha="center", va="center",
                color="white" if matriz_confusao[i, j] > matriz_confusao.max() / 2 else "black"
            )

    caminho = os.path.join(
        REPORT_DIR, f"confusion_matrix_{nome_modelo.lower().replace(' ', '_')}.png"
    )
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info(f"Matriz de confusão gerada para {nome_modelo}")


def plot_regression_actual_vs_predicted(y_teste, y_pred):

    plt.figure(figsize=(7, 6))

    plt.scatter(y_teste, y_pred, alpha=0.5, color="#4C72B0", s=15)

    limite_min = min(np.min(y_teste), np.min(y_pred))
    limite_max = max(np.max(y_teste), np.max(y_pred))
    plt.plot([limite_min, limite_max], [limite_min, limite_max], color="crimson", linestyle="--")

    plt.title("Regressão Múltipla: Valores Reais vs. Previstos")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.grid(True, alpha=0.3)

    caminho = os.path.join(REPORT_DIR, "regression_actual_vs_predicted.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=120)
    plt.close()

    logging.info("Gráfico de regressão (real vs. previsto) gerado")