# Explicação dos Códigos

## Código 1 -- Média, Variância, Desvio Padrão e Z-score

Este código calcula manualmente medidas estatísticas básicas para o
conjunto de dados:

``` python
[1, 2, 2, 10]
```

### Etapas

1.  Calcula a **média** (`np.mean`).
2.  Calcula os **desvios** (`x - média`).
3.  Eleva os desvios ao quadrado.
4.  Calcula a **variância** como a média dos desvios ao quadrado.
5.  Calcula o **desvio padrão** (`sqrt(variância)`).
6.  Calcula o **Z-score** de cada valor:

\[ Z=`\frac{x-\bar{x}}{SD}`{=tex} \]

7.  Organiza tudo em um `DataFrame` e imprime os resultados.

O Z-score mostra quantos desvios padrão cada observação está distante da
média.

------------------------------------------------------------------------

## Código 2 -- Teorema do Limite Central

Este código demonstra o **Teorema do Limite Central (TLC)**.

### Funcionamento

-   Gera uma população de 10.000 atrasos usando uma distribuição
    **exponencial**, que é assimétrica.
-   Extrai **10.000 amostras**, cada uma com 100 elementos.
-   Calcula a média de cada amostra.
-   Calcula:
    -   média da população;
    -   média das médias amostrais;
    -   desvio padrão teórico das médias
        ((`\sigma`{=tex}/`\sqrt{n}`{=tex}));
    -   desvio padrão observado das médias.

### Objetivo

Mostrar que, mesmo partindo de uma população assimétrica, a distribuição
das médias amostrais tende a ser aproximadamente normal quando o tamanho
da amostra é grande.

------------------------------------------------------------------------

## Código 3 -- Simulação de Teste de Hipótese

Este código utiliza uma distribuição binomial para simular 10.000
experimentos de lançamento de 30 moedas assumindo:

-   Hipótese nula (**H₀**): a moeda é justa (`p = 0,5`).

### Etapas

1.  Simula 10.000 experimentos.
2.  Calcula o intervalo empírico central de 95% usando os percentis de
    2,5% e 97,5%.
3.  Calcula o **p-valor** para observar 22 ou mais caras.
4.  Compara o p-valor com 0,05.

Se:

-   `p < 0,05` → rejeita H₀ (evidência de viés).
-   `p >= 0,05` → não há evidências suficientes para rejeitar H₀.
