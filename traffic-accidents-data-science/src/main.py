import os

from extract.extractor import extract_data
from transform.cleaner import clean_data
from utils.columns import resolve_column

from inference.bootstrap import run_bootstrap_analysis
from inference.ab_testing import run_ab_test

from models.regression import fit_multiple_regression
from models.machine_learning import run_classification
from models.unsupervised import run_unsupervised_analysis

import visualize


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


COL_MORTOS = "MORTOS"
COL_FERIDOS = "FERIDOS"
COL_KM = "KM"
COL_ANO = "ANO"
COL_LATITUDE = "LATITUDE"
COL_LONGITUDE = "LONGITUDE"
COLUNA_BOOTSTRAP = COL_KM
COLUNA_GRUPO_AB = COL_MORTOS
COLUNA_METRICA_AB = COL_FERIDOS
COLUNA_Y_REGRESSAO = COL_FERIDOS
COLUNAS_X_REGRESSAO = [COL_KM, COL_MORTOS]
COLUNA_GRAVIDADE_CLF = COL_MORTOS
COLUNAS_FEATURES_CLF = [COL_KM, COL_FERIDOS, COL_LATITUDE, COL_LONGITUDE, COL_ANO]
COLUNAS_FEATURES_NAO_SUPERVISIONADO = [
    COL_KM, COL_FERIDOS, COL_MORTOS, COL_LATITUDE, COL_LONGITUDE
]


def _titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def executar_parte1():

    _titulo("PARTE 1 — ETL (Extração, Limpeza e Visualização Descritiva)")

    print("Extraindo dados...")
    df = extract_data(arquivo)

    print("Tratando dados...")
    df_limpo = clean_data(df)

    print("Salvando arquivo final...")
    df_limpo.to_csv(saida, index=False, encoding="utf-8")

    print("Gerando visualizações descritivas...")
    visualize.generate_visualizations(df_limpo)

    return df_limpo


def executar_bootstrap(df):

    _titulo("PARTE 2.1 — BOOTSTRAP E INTERVALOS DE CONFIANÇA")

    coluna = resolve_column(df, COLUNA_BOOTSTRAP)

    resultado = run_bootstrap_analysis(df, coluna, n_boot=2000, confianca=95.0)

    est = resultado["estatisticas"]
    print(f"Variável analisada: {coluna}")
    print(f"N = {est['n']} | Média (X̄) = {est['media']:.4f} | Desvio Padrão (s) = {est['desvio_padrao']:.4f}")
    print(f"IC 95% Bootstrap (percentil):     [{resultado['ic_bootstrap'][0]:.4f}, {resultado['ic_bootstrap'][1]:.4f}]")
    print(f"IC 95% Paramétrico (Normal/TCL):  [{resultado['ic_parametrico'][0]:.4f}, {resultado['ic_parametrico'][1]:.4f}]")
    print(f"Diagnóstico TCL: {resultado['diagnostico_tcl']['parecer']}")

    visualize.plot_bootstrap_distribution(resultado)

    return resultado


def executar_ab_test(df):

    _titulo("PARTE 2.2 — TESTE A/B (TESTE DE PERMUTAÇÃO)")

    coluna_grupo = resolve_column(df, COLUNA_GRUPO_AB)
    coluna_metrica = resolve_column(df, COLUNA_METRICA_AB)

    print("H0: mu_A - mu_B = 0 (não há diferença entre os grupos)")
    print("H1: mu_A - mu_B != 0 (existe diferença entre os grupos), teste bicaudal")
    print("alpha = 0.05")

    resultado = run_ab_test(
        df, group_col=coluna_grupo, metric_col=coluna_metrica,
        n_perm=2000, alpha=0.05,
    )

    print(f"Grupo A (graves/fatais): n={resultado['n_grupo_a']} | média {coluna_metrica} = {resultado['media_grupo_a']:.4f}")
    print(f"Grupo B (leves/sem vítimas): n={resultado['n_grupo_b']} | média {coluna_metrica} = {resultado['media_grupo_b']:.4f}")
    print(f"Diferença observada: {resultado['diferenca_observada']:.4f}")
    print(f"p-valor empírico (bicaudal): {resultado['p_valor']:.4f}")
    print(resultado["conclusao"])

    visualize.plot_permutation_distribution(resultado)

    return resultado


def executar_regressao(df):

    _titulo("PARTE 2.3 — REGRESSÃO LINEAR MÚLTIPLA")

    y_col = resolve_column(df, COLUNA_Y_REGRESSAO)
    x_cols = [resolve_column(df, c) for c in COLUNAS_X_REGRESSAO]

    resultado = fit_multiple_regression(df, y_col=y_col, x_cols=x_cols)

    print(f"Y = {y_col} | X = {x_cols}")
    for nome, valor in resultado["coeficientes"].items():
        print(f"  {nome} = {valor:.6f}")
    print(f"R² (teste) = {resultado['r2']:.4f}")
    print(f"RMSE (teste) = {resultado['rmse']:.4f}")

    visualize.plot_regression_actual_vs_predicted(
        resultado["y_teste"], resultado["y_pred"]
    )

    return resultado


def executar_classificacao(df):

    _titulo("PARTE 2.3 — CLASSIFICAÇÃO BINÁRIA (LOGÍSTICA vs. KNN)")

    coluna_gravidade = resolve_column(df, COLUNA_GRAVIDADE_CLF)
    colunas_features = [resolve_column(df, c) for c in COLUNAS_FEATURES_CLF]

    resultado = run_classification(
        df, feature_cols=colunas_features, severity_col=coluna_gravidade,
        cv_folds=5,
    )

    for nome_modelo in ["regressao_logistica", "knn"]:
        r = resultado[nome_modelo]
        nome_exibicao = "Regressão Logística" if nome_modelo == "regressao_logistica" else "KNN"

        print(f"\n--- {nome_exibicao} ---")
        print(f"Melhores hiperparâmetros (GridSearchCV): {r['melhores_parametros']}")
        print(f"F1-Score médio (Validação Cruzada): {r['melhor_f1_cv']:.4f}")
        print(f"Acurácia (teste): {r['acuracia']:.4f}")
        print(f"Precisão (teste): {r['precisao']:.4f}")
        print(f"Sensibilidade/Recall (teste): {r['sensibilidade']:.4f}")
        print(f"F1-Score (teste): {r['f1_score']:.4f}")
        print(f"Matriz de Confusão:\n{r['matriz_confusao']}")

        visualize.plot_confusion_matrix(r["matriz_confusao"], nome_exibicao)

    return resultado


def executar_nao_supervisionado(df):

    _titulo("PARTE 2.4 — PCA E K-MEANS (APRENDIZADO NÃO SUPERVISIONADO)")

    colunas_features = [
        resolve_column(df, c) for c in COLUNAS_FEATURES_NAO_SUPERVISIONADO
    ]

    resultado = run_unsupervised_analysis(
        df, feature_cols=colunas_features, k_min=1, k_max=10
    )

    variancia = resultado["pca"]["variancia_explicada"]
    print(f"Features utilizadas: {colunas_features}")
    print(f"Variância explicada PC1: {variancia[0] * 100:.2f}% | PC2: {variancia[1] * 100:.2f}%")
    print(f"Variância acumulada (PC1+PC2): {resultado['pca']['variancia_acumulada'] * 100:.2f}%")
    print(f"k escolhido pelo Método do Cotovelo: {resultado['k_escolhido']}")
    print(f"Inércia final do K-Means: {resultado['kmeans']['inertia']:.2f}")

    visualize.plot_elbow_curve(resultado["cotovelo"], resultado["k_escolhido"])
    visualize.plot_pca_projection(resultado["pca"])
    visualize.plot_kmeans_clusters(resultado["pca"], resultado["kmeans"])

    return resultado


def main():

    df_limpo = executar_parte1()

    executar_bootstrap(df_limpo)

    executar_ab_test(df_limpo)

    executar_regressao(df_limpo)

    executar_classificacao(df_limpo)

    executar_nao_supervisionado(df_limpo)

    _titulo("PIPELINE COMPLETO (PARTE 1 + PARTE 2) EXECUTADO COM SUCESSO!")


if __name__ == "__main__":

    main()
