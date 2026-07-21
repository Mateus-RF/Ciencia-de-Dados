# Explicação dos Códigos

## Código 1 -- Função reutilizável para Intervalo de Confiança

O código cria uma função chamada `calcular_ic_api()` para calcular o
intervalo de confiança da média de um conjunto de dados.

### O que a função faz

1.  Calcula o tamanho da amostra (`n`).
2.  Calcula a média amostral.
3.  Calcula o desvio padrão amostral (`ddof=1`).
4.  Calcula o erro padrão:

\[ SE=`\frac{s}{\sqrt{n}}`{=tex} \]

5.  Obtém o valor crítico da distribuição Normal.
6.  Calcula a margem de erro.
7.  Retorna:
    -   média;
    -   intervalo de confiança;
    -   margem de erro.

Depois disso o código gera 60 latências de API e calcula automaticamente
o IC de 95%.

------------------------------------------------------------------------

## Código 2 -- Cálculo manual do Intervalo de Confiança

Nesse exemplo os valores da amostra já são conhecidos.

Dados:

-   n = 50
-   média = 3.2
-   desvio padrão = 1.74

O programa calcula:

1.  Erro padrão.
2.  Valor crítico z para 95%.
3.  Margem de erro.
4.  Limites inferior e superior do intervalo de confiança.

É uma implementação direta da fórmula do IC para a média.

------------------------------------------------------------------------

## Código 3 -- Regra Empírica da Distribuição Normal

Esse código utiliza a função `stats.norm.cdf()` para calcular a área sob
a curva Normal entre:

-   média ± 1 desvio padrão;
-   média ± 2 desvios padrão;
-   média ± 3 desvios padrão.

Os resultados obtidos são aproximadamente:

-   ±1 SD → 68,27%
-   ±2 SD → 95,45%
-   ±3 SD → 99,73%

Os valores correspondem à conhecida Regra 68--95--99,7 da
distribuição Normal.

------------------------------------------------------------------------

## Código 4 -- Comparando diferentes níveis de confiança

O código mostra como o intervalo de confiança muda conforme o nível
de confiança aumenta.

São calculados os níveis:

-   90%
-   95%
-   98%
-   99%

Para cada nível o programa calcula:

-   valor crítico z;
-   margem de erro;
-   intervalo de confiança.

### Observação

Quanto maior o nível de confiança:

-   maior será o valor crítico (z\*);
-   maior será a margem de erro;
-   mais largo será o intervalo de confiança.

Acontece porque é necessário um intervalo maior para aumentar a
confiança de que o verdadeiro parâmetro esteja dentro dele.


