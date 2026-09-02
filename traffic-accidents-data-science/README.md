# Análise de Acidentes de Trânsito no Brasil

Equipe: Alice Maria de Lima Melo, Ana Rosimeire Ferreira dos Santos e Mateus Ribeiro Ferreira
## 1. Descrição do Projeto

Este projeto tem como objetivo desenvolver um pipeline de dados aplicado à análise de acidentes de trânsito no Brasil, utilizando técnicas de Ciência de Dados para realizar a ingestão, tratamento, análise exploratória e visualização dos dados.

O pipeline foi desenvolvido buscando garantir a reprodutibilidade das etapas de processamento, desde a leitura dos dados brutos até a geração de uma base final limpa e preparada para futuras análises estatísticas e aplicações de modelos preditivos.

O tema escolhido pertence à área de **Logística e Segurança Viária**, utilizando dados públicos disponibilizados na plataforma Kaggle.

---

# 2. Fonte dos Dados

O conjunto de dados utilizado foi:

**Traffic Accidents Brazil**

Fonte:
Kaggle - Rafael Borges Graunke

Dataset principal:

```
datatran2007-2022.csv
```

O conjunto apresenta registros de acidentes de trânsito ocorridos no Brasil entre os anos de 2007 e 2022, contendo informações relacionadas à localização, características da ocorrência, quantidade de envolvidos, vítimas e condições do acidente.

---

# 3. Estrutura do Pipeline

O projeto foi dividido nas seguintes etapas:

```
src/
│
├── extract/
│   └── extractor.py
│
├── transform/
│   └── cleaner.py
│
├── inference/
│   ├── bootstrap.py        (Parte 2 — Bootstrap e Intervalos de Confiança)
│   └── ab_testing.py       (Parte 2 — Teste de Permutação / Teste A-B)
│
├── models/
│   ├── regression.py       (Parte 2 — Regressão Linear Múltipla)
│   ├── machine_learning.py (Parte 2 — Classificação: Logística vs. KNN)
│   └── unsupervised.py     (Parte 2 — PCA e K-Means)
│
├── utils/
│   └── columns.py          (resolução tolerante de nomes de colunas)
│
├── visualize.py            (script consolidado de geração de gráficos)
│
└── main.py                 (orquestrador de todo o pipeline: ETL + Inferência + Modelagem)
```

### extractor.py

Responsável pela camada de ingestão dos dados.

Realiza:

- leitura automatizada do arquivo CSV;
- tratamento de erros relacionados ao caminho do arquivo;
- identificação das dimensões iniciais da base;
- carregamento dos dados brutos para processamento.


### cleaner.py

Responsável pela transformação e tratamento estatístico.

Realiza:

- padronização de variáveis categóricas;
- tratamento de valores ausentes;
- conversão de tipos de dados;
- identificação de valores extremos utilizando o método IQR.


### visualize.py

Responsável pela geração de **todas** as visualizações do projeto (script consolidado), tanto da Parte 1 (EDA) quanto da Parte 2 (inferência e modelagem).

São produzidos gráficos para análise:

- evolução temporal dos acidentes;
- distribuição dos acidentes por estado;
- relação entre variáveis numéricas;
- distribuição Bootstrap da média com os dois métodos de IC (Parte 2);
- distribuição do Teste de Permutação sob H0 (Parte 2);
- curva do cotovelo, projeção PCA e clusters K-Means (Parte 2);
- matriz de confusão dos classificadores e reta real-vs-previsto da regressão (bônus).

### inference/bootstrap.py e inference/ab_testing.py

Camada de **Inferência Estatística** (Parte 2): reamostragem Bootstrap, cálculo dos dois tipos de Intervalo de Confiança, definição de grupos A/B e Teste de Permutação com p-valor empírico.

### models/regression.py, models/machine_learning.py e models/unsupervised.py

Camada de **Modelagem** (Parte 2): regressão linear múltipla, classificação binária (Regressão Logística vs. KNN com GridSearchCV) e aprendizado não supervisionado (PCA + K-Means com Método do Cotovelo).

---

# 4. Camada de Ingestão, Amostragem e Viés

## População-alvo

A população-alvo ideal deste estudo corresponde a todos os acidentes de trânsito ocorridos no território brasileiro durante o período analisado.

Essa população representa todos os eventos reais de acidentes, independentemente de terem sido registrados oficialmente ou não.

---

## Estrutura de Acesso (Access Frame)

A estrutura de acesso disponível corresponde aos registros de acidentes coletados pelos sistemas oficiais responsáveis pelo monitoramento das ocorrências de trânsito.

Portanto, o dataset representa apenas os acidentes que foram registrados e disponibilizados na fonte utilizada.

---

## Possíveis Viéses de Seleção

O dataset pode apresentar viés de seleção devido à possibilidade de subnotificação dos acidentes.

Alguns fatores que podem influenciar esse problema:

- acidentes de menor gravidade podem não ser registrados;
- diferenças na eficiência de coleta entre regiões;
- ausência de informações sobre ocorrências sem atendimento oficial.

Dessa forma, os dados disponíveis podem não representar perfeitamente todos os acidentes que realmente ocorreram no país.

---

# 5. Tratamento e Análise Exploratória dos Dados (EDA)

A etapa de tratamento teve como objetivo melhorar a qualidade dos dados e reduzir inconsistências que poderiam prejudicar análises posteriores.

Foram aplicados os seguintes procedimentos:

## Tratamento de Valores Ausentes

Os valores ausentes encontrados nas variáveis numéricas foram tratados utilizando a substituição pela mediana.

A escolha dessa estratégia ocorreu porque a mediana apresenta menor sensibilidade a valores extremos quando comparada à média.

### Impacto no viés e variância

A imputação evita a remoção de registros, mantendo maior quantidade de informações disponíveis.

Entretanto, essa abordagem pode reduzir a variabilidade natural dos dados, diminuindo a variância do conjunto final.

Consequentemente, existe o risco de suavização de características reais presentes na população original.

---

# 6. Identificação de Outliers utilizando IQR

Para identificação automática de valores extremos foi utilizado o método do Intervalo Interquartil (IQR).

O cálculo utilizado foi:

\[
IQR = Q3 - Q1
\]

Foram definidos os limites:

\[
Limite\ inferior = Q1 - 1.5 \times IQR
\]

\[
Limite\ superior = Q3 + 1.5 \times IQR
\]


Valores encontrados fora desses limites foram considerados possíveis outliers.

A utilização desse método permite reduzir distorções estatísticas causadas por valores extremos, mantendo uma abordagem baseada em critérios matemáticos.

### Ressalva importante: colunas excluídas do filtro IQR

As colunas **MORTOS**, **FERIDOS** e **ANO** foram **excluídas** da remoção de outliers por IQR.

MORTOS e FERIDOS são contagens de vítimas fortemente assimétricas — a grande maioria dos registros vale 0. Nesses casos, Q1 = Q3 = 0, e o método IQR classificaria **qualquer** acidente com vítima como outlier, removendo do dataset exatamente os acidentes graves/fatais que são a população de maior interesse para a análise de severidade (Seção 15 — Teste A/B). Manter esse filtro ativo introduziria um viés de seleção severo, "limpando" para fora do dataset o próprio fenômeno em estudo. ANO foi excluído por ser um identificador discreto de período, não uma medida sujeita a erro de mensuração.

Essa correção evidencia, na prática, um dos riscos discutidos na Seção 4: procedimentos estatísticos aplicados sem considerar a natureza semântica de cada variável podem gerar viés de seleção mesmo em pipelines aparentemente bem construídos.

---

# 7. Dicionário de Dados

| Variável | Descrição | Tipo Estatístico |
|---|---|---|
| ANO | Ano de ocorrência do acidente | Discreta |
| UF | Estado onde ocorreu o acidente | Categórica |
| MUNICIPIO | Município da ocorrência | Categórica |
| BR | Rodovia federal relacionada ao acidente | Categórica |
| KM | Quilometragem da ocorrência | Contínua |
| FERIDOS | Quantidade de pessoas feridas | Discreta |
| MORTOS | Quantidade de vítimas fatais | Discreta |
| LATITUDE | Coordenada geográfica de latitude | Contínua |
| LONGITUDE | Coordenada geográfica de longitude | Contínua |
| CAUSA_ACIDENTE | Causa registrada para o acidente | Categórica |
| TIPO_ACIDENTE | Classificação do acidente | Categórica |

---

# 8. Análise de Domínio e Inferência Causal

## Hipótese analisada

Uma hipótese relacionada ao domínio estudado é:

**"Condições climáticas adversas podem aumentar a quantidade de acidentes de trânsito."**

---

## a) Correlação não representa causalidade

Mesmo que seja encontrada uma forte correlação matemática entre duas variáveis, isso não significa necessariamente que uma variável seja responsável pela alteração da outra.

A correlação apenas indica que existe uma associação entre os valores observados.

Por exemplo, pode existir uma relação entre dias chuvosos e maior quantidade de acidentes, porém essa relação pode ser influenciada por outros fatores externos.

Portanto, somente a análise de correlação não é suficiente para afirmar um efeito causal.

---

# 9. Variáveis de Confusão (Confounders)

## Volume de tráfego

A quantidade de veículos circulando em uma determinada via pode influenciar diretamente o número de acidentes.

Rodovias com maior fluxo apresentam maior probabilidade de ocorrências, independentemente das condições climáticas.

---

## Condições da infraestrutura

A qualidade da estrada, iluminação, sinalização e conservação da via também podem influenciar a ocorrência de acidentes.

Uma rodovia com problemas estruturais pode apresentar maior número de acidentes mesmo em condições climáticas normais.

---

# 10. Cenário Ideal utilizando Ceteris Paribus

O princípio de **Ceteris Paribus** busca analisar o efeito de uma variável mantendo todas as outras condições constantes.

Um cenário ideal para avaliar o impacto das condições climáticas seria comparar acidentes:

- na mesma rodovia;
- no mesmo horário;
- com volume de veículos semelhante;
- com condições equivalentes de infraestrutura.

Dessa forma, seria possível reduzir a influência de fatores externos e estimar melhor o efeito específico das condições climáticas sobre os acidentes.

---

# 11. Visualização Científica

As visualizações produzidas seguem princípios de integridade visual, garantindo:

- identificação correta dos eixos;
- descrição das métricas apresentadas;
- utilização de escalas proporcionais;
- ausência de distorções gráficas.

Os gráficos gerados permitem observar:

- evolução dos acidentes ao longo dos anos;
- distribuição dos acidentes entre estados;
- relação entre variáveis numéricas.

---

# 12. Execução do Projeto

## Instalação das dependências

Executar:
```bash
python -m venv venv
```

```bash
cd traffic-accidents-data-science
```

```bash
pip install -r requirements.txt
```

---

## Execução do pipeline completo

Executar:

```bash
python src/main.py
```

---

Após a execução, o pipeline gera automaticamente:

## Dataset tratado

```
data/processed/dados_limpos_final.csv
```

## Gráficos descritivos (Parte 1) e complementares (bônus)

```
reports/figures/
```

## Gráficos de inferência e modelagem (Parte 2)

Salvos na **raiz do projeto**, conforme especificação da Avaliação Prática 2:

```
distribuicao_bootstrap.png
distribuicao_permutacao.png
curva_cotovelo_kmeans.png
clusters_kmeans.png
pca_projecao.png
```

---

# 13. Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- Matplotlib

---

# 15. Estimação de Parâmetros e Bootstrap

## Variável analisada

Foi escolhida a variável contínua **KM** (quilometragem da rodovia onde ocorreu o acidente) como variável-chave para a estimação Bootstrap.

## Metodologia

1. Cálculo da Média Amostral (X̄) e do Desvio Padrão Amostral (s) observados na base tratada.
2. Reamostragem Bootstrap não-paramétrica com **2.000 réplicas com reposição**, gerando a distribuição empírica da média.
3. Construção de dois Intervalos de Confiança (IC) de 95% para a média populacional:
   - **Método Bootstrap (não-paramétrico):** percentis 2,5% e 97,5% da distribuição empírica das médias reamostradas.
   - **Método Paramétrico Tradicional:** aproximação Normal via TCL, IC95% = X̄ ± 1,96 · (s / √N).

## Comparação entre as metodologias

Nos testes de validação do pipeline (dados sintéticos com N ≈ 3.000), os dois métodos produziram intervalos praticamente coincidentes (diferença de centésimos entre os limites). Isso é esperado: com uma amostra grande e uma distribuição original razoavelmente simétrica, tanto a aproximação Normal quanto o Bootstrap convergem para o mesmo resultado. A vantagem do Bootstrap aparece justamente quando essas condições não se sustentam — variáveis fortemente assimétricas, com outliers, ou com N pequeno — situações em que ele não depende da suposição de Normalidade.

## As condições do TCL se aplicam?

O diagnóstico automático implementado em `bootstrap.py` (`check_clt_conditions`) avalia o tamanho amostral (regra prática N ≥ 30) e a assimetria (skewness) da variável original. Para KM, com milhares de registros e assimetria moderada, as condições do TCL são consideradas satisfeitas, o que justifica a boa concordância entre os dois métodos de IC observada acima. **Ao rodar com o dataset real**, o `main.py` imprime esse parecer automaticamente para a variável configurada.

---

# 16. Teste de Hipóteses e Teste A/B (Teste de Permutação)

## Cenário e grupos de teste

Dentro da área de Logística e Segurança Viária, os dados foram segmentados em:

- **Grupo A** — acidentes **graves/fatais** (`MORTOS > 0`);
- **Grupo B** — acidentes **leves/sem vítimas** (`MORTOS == 0`).

**Métrica comparada:** número de **FERIDOS** por ocorrência. O dataset da Parte 1 não possui velocidade ou distância de frenagem registradas (sugestões originais do enunciado), portanto foi adotada a métrica de vítimas feridas como proxy contínua/discreta de severidade, disponível diretamente no dicionário de dados da Seção 7 e não utilizada para definir os próprios grupos.

## Hipóteses e significância

- **H0:** μ_A − μ_B = 0 (não há diferença na média de feridos entre acidentes graves/fatais e leves/sem vítimas).
- **H1:** μ_A − μ_B ≠ 0 (existe diferença), teste **bicaudal**.
- **α = 0,05**.

## Metodologia

1. Cálculo da estatística observada: diferença entre as médias amostrais dos dois grupos (X̄_A − X̄_B).
2. Teste de Permutação com **2.000 iterações**: os rótulos de grupo são embaralhados e a diferença de médias é recalculada a cada iteração, simulando a distribuição da estatística sob H0.
3. Cálculo do **p-valor empírico bicaudal**: proporção de diferenças permutadas cujo valor absoluto é maior ou igual ao valor absoluto da diferença observada.

## Conclusão e significado prático

Nos testes de validação, acidentes graves/fatais apresentaram, em média, número de feridos consideravelmente maior que os leves/sem vítimas, e o p-valor empírico ficou abaixo de 0,05 — **rejeitando H0**. Ao rodar o pipeline com o dataset real, `main.py` imprime automaticamente as médias de cada grupo, a diferença observada, o p-valor e a conclusão formal.

Do ponto de vista de negócio/segurança viária: se o resultado real confirmar essa rejeição de H0, isso reforça que acidentes classificados como graves/fatais tendem a envolver sistematicamente mais pessoas feridas — um indicativo indireto de que a gravidade de um acidente (mortes) está associada a um padrão mais amplo de dano (mais feridos), e não é um evento isolado e independente da magnitude do acidente.

---

# 17. Regressão Linear Múltipla

## Especificação do modelo

- **Y (variável resposta):** FERIDOS.
- **X1:** KM.
- **X2:** MORTOS.

## Interpretação dos coeficientes (Ceteris Paribus)

- **β0 (intercepto):** número esperado de feridos quando KM = 0 e MORTOS = 0.
- **β1 (KM):** variação esperada no número de feridos para cada quilômetro adicional na rodovia, mantendo MORTOS constante. Nos testes de validação, esse coeficiente ficou próximo de zero, sugerindo que a posição quilométrica isoladamente tem pouca relação linear com o número de feridos.
- **β2 (MORTOS):** variação esperada no número de feridos para cada vítima fatal adicional, mantendo KM constante. Esse coeficiente foi consistentemente positivo e de magnitude relevante, indicando que acidentes com mais mortes tendem a registrar também mais feridos — coerente com a ideia de que ambos refletem a mesma causa subjacente (gravidade/violência do impacto).

## Qualidade do ajuste

O modelo reporta **R²** e **RMSE** calculados sobre um conjunto de teste (20% dos dados, nunca visto no treino). Um R² baixo é esperado neste caso: o número de feridos depende de diversos fatores não incluídos no modelo (velocidade, tipo de colisão, número de veículos envolvidos, uso de cinto de segurança, etc.), portanto duas variáveis apenas (KM e MORTOS) explicam somente uma fração da variabilidade total — o que reforça a necessidade de cautela ao interpretar causalidade a partir deste modelo simples.

---

# 18. Modelagem de Classificação (Regressão Logística vs. KNN)

## Tarefa

Classificação binária: acidente **grave** (MORTOS > 0 → 1) vs. **não grave** (MORTOS == 0 → 0), a partir de variáveis descritivas da ocorrência (KM, FERIDOS, LATITUDE, LONGITUDE, ANO) — nenhuma delas é a própria variável usada para derivar o alvo.

## Pipeline

1. Pré-processamento com `StandardScaler` (padronização) dentro de um `Pipeline` do Scikit-Learn, evitando vazamento de informação entre treino e teste.
2. Otimização de hiperparâmetros via **GridSearchCV** com Validação Cruzada (5 folds), otimizando F1-Score:
   - Regressão Logística: busca sobre o parâmetro de regularização `C`.
   - KNN: busca sobre o número de vizinhos `n_neighbors` e o tipo de ponderação `weights`.
3. Avaliação final no conjunto de teste (nunca usado na busca de hiperparâmetros): Matriz de Confusão, Acurácia, Precisão, Sensibilidade (Recall) e F1-Score.

## Discussão

Este é um problema de classes desbalanceadas (acidentes fatais são minoria), por isso a **Sensibilidade (Recall)** e o **F1-Score** são métricas mais informativas do que a Acurácia isolada — um classificador que sempre prevê "não grave" atingiria acurácia alta, mas seria inútil na prática (Recall próximo de zero). Nos testes de validação, o KNN otimizado apresentou Recall e F1-Score superiores aos da Regressão Logística nesse cenário desbalanceado, ainda que com acurácia global ligeiramente menor — uma troca (trade-off) relevante do ponto de vista de segurança viária, onde deixar de identificar um acidente potencialmente grave (falso negativo) tende a ser mais custoso do que um falso positivo.

---

# 19. Aprendizado Não Supervisionado (PCA e K-Means)

## Redução de Dimensionalidade (PCA)

As variáveis numéricas KM, FERIDOS, MORTOS, LATITUDE e LONGITUDE foram padronizadas (`StandardScaler`) e projetadas nos dois primeiros Componentes Principais, obtidos via Decomposição em Valores Singulares (SVD) — fundamento matemático do PCA. `main.py` imprime a variância explicada por cada componente e a variância acumulada pelos dois primeiros. Quanto maior essa variância acumulada, mais fielmente o gráfico 2D representa a estrutura multidimensional original dos dados; quando ela é baixa, a projeção deve ser interpretada como uma simplificação útil para visualização, mas parcial.

## Clusterização (K-Means) e Método do Cotovelo

O número ótimo de clusters (k) foi determinado pelo **Método do Cotovelo**: a inércia (soma dos quadrados intra-cluster) é calculada para k de 1 a 10, e o pipeline seleciona automaticamente o ponto de maior curvatura da curva (o "cotovelo") como k ideal — o ponto a partir do qual aumentar k passa a reduzir a inércia de forma marginal, sem justificar a complexidade adicional de mais grupos.

## Significado prático dos clusters

Cada cluster tende a agrupar acidentes com combinações semelhantes de localização geográfica (latitude/longitude), quilometragem e severidade (feridos/mortos). Na prática, isso pode revelar, por exemplo, regiões ou trechos rodoviários com padrões de severidade sistematicamente diferentes — informação que pode orientar a priorização de fiscalização, sinalização ou manutenção viária em rodovias/trechos específicos, mesmo sem que se tenha estabelecido uma relação causal explícita.

---

# 20. Inferência Causal e Tomada de Decisão

## a) Correlação vs. Causalidade nos modelos supervisionados

Os modelos de regressão e classificação implementados nas Seções 17 e 18 identificam **associações estatísticas** entre variáveis (por exemplo, entre MORTOS e FERIDOS), não relações causais. Um coeficiente positivo e significativo entre MORTOS e FERIDOS não implica que um "causa" o outro; é mais plausível que ambos sejam consequências de uma causa comum — a violência/energia do impacto do acidente.

## b) Variáveis de confusão (confounders)

Como já discutido na Seção 9, **volume de tráfego** e **condições de infraestrutura** são confounders plausíveis: rodovias mais movimentadas ou com pior infraestrutura tendem a ter mais acidentes e, potencialmente, acidentes mais graves, independentemente de outros fatores incluídos nos modelos. Isso viola a hipótese de independência estatística necessária para interpretar os coeficientes da regressão múltipla (Seção 17) como efeitos causais "limpos": parte do efeito atribuído a KM ou MORTOS pode, na verdade, refletir a influência não observada dessas variáveis de confusão.

## c) Tomada de decisão operacional

Com base no comportamento dos clusters (Seção 19) e nas previsões do classificador (Seção 18), uma decisão operacional razoável seria priorizar ações de fiscalização e melhoria de infraestrutura nos trechos/rodovias associados aos clusters de maior severidade média, e usar o classificador (ajustado para maximizar Recall, dado o custo assimétrico de um falso negativo) como um sistema de triagem para direcionar recursos de resposta a emergência a ocorrências com maior probabilidade de gravidade — sempre reconhecendo que essas são recomendações baseadas em correlação e padrões observados, não em relações causais comprovadas.

---

# 21. Considerações Finais

O desenvolvimento deste pipeline possibilitou aplicar conceitos fundamentais de Ciência de Dados, incluindo ingestão de dados, tratamento estatístico, análise exploratória, identificação de outliers e interpretação científica dos resultados (Parte 1), além de inferência estatística computacional, modelagem preditiva supervisionada, aprendizado não supervisionado e raciocínio causal (Parte 2).

Além do processamento computacional, foram consideradas limitações relacionadas à qualidade dos dados, possíveis vieses de seleção — incluindo um viés introduzido pelo próprio pipeline de limpeza e corrigido nesta etapa (Seção 6) — e cuidados necessários para evitar interpretações causais incorretas a partir de modelos estatísticos e de aprendizado de máquina.

> **Nota sobre os números apresentados neste README:** os valores citados nas Seções 15 a 19 refletem uma execução de validação do pipeline com dados sintéticos, usada apenas para garantir que o código funciona de ponta a ponta. Ao rodar `python src/main.py` com o dataset real (`datatran2007-2022.csv`), todos os números, gráficos e conclusões serão recalculados automaticamente a partir dos dados verdadeiros.