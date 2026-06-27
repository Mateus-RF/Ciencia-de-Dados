# Análise de Acidentes de Trânsito no Brasil

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
├── visualize.py
│
└── main.py
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

Responsável pela geração das visualizações.

São produzidos gráficos para análise:

- evolução temporal dos acidentes;
- distribuição dos acidentes por estado;
- relação entre variáveis numéricas.

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

## Gráficos

```
reports/figures/
```

---

# 13. Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn


---

# 14. Considerações Finais

O desenvolvimento deste pipeline possibilitou aplicar conceitos fundamentais de Ciência de Dados, incluindo ingestão de dados, tratamento estatístico, análise exploratória, identificação de outliers e interpretação científica dos resultados.

Além do processamento computacional, foram consideradas limitações relacionadas à qualidade dos dados, possíveis vieses de seleção e cuidados necessários para evitar interpretações causais incorretas.