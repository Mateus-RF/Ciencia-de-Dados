# Explicação dos Códigos

## Código 1 -- Amostragem com NumPy e Pandas

Este código demonstra diferentes formas de realizar amostragem.

### 1. Criação da população

``` python
populacao = np.array(['Aluno_A', 'Aluno_B', 'Aluno_C', 'Aluno_D', 'Aluno_E'])
```

Cria uma população com cinco alunos.

### 2. Amostragem com reposição

``` python
np.random.choice(populacao, size=3, replace=True)
```

Sorteia 3 alunos permitindo repetição.

### 3. Amostragem sem reposição

``` python
np.random.choice(populacao, size=3, replace=False)
```

Sorteia 3 alunos sem repetir nenhum.

### 4. Criando um DataFrame

``` python
df_users = pd.DataFrame({'id': range(100), 'ativo': np.random.choice([0, 1], 100)})
```

Cria uma tabela com 100 usuários e uma coluna indicando se o usuário
está ativo (1) ou inativo (0).

### 5. Amostrando linhas do DataFrame

``` python
amostra = df_users.sample(n=5, random_state=42)
```

Seleciona aleatoriamente 5 linhas. O `random_state=42` garante que o
mesmo resultado seja obtido sempre que o código for executado.

------------------------------------------------------------------------

## Código 2 -- Simulação de Probabilidade

Este código utiliza simulação para estimar probabilidades.

### 1. Definindo a semente

``` python
np.random.seed(42)
```

Faz com que os resultados aleatórios possam ser reproduzidos.

### 2. Simulando lançamentos de moedas

``` python
m1 = np.random.choice(['H', 'T'], size=10000)
m2 = np.random.choice(['H', 'T'], size=10000)
```

Simula 10.000 lançamentos de duas moedas, onde: - `H` = Cara (Heads) -
`T` = Coroa (Tails)

### 3. Probabilidade de pelo menos uma cara

``` python
p_pelo_menos_uma_cara = np.mean((m1 == 'H') | (m2 == 'H'))
```

Conta a proporção de experimentos em que pelo menos uma das moedas
resultou em cara. O valor esperado é aproximadamente **75% (3/4)**.

### 4. Probabilidade teórica sem reposição

``` python
p_yx = (1/3) * (1/2)
```

Calcula uma probabilidade teórica utilizando a regra da multiplicação
para eventos dependentes (sem reposição): - primeira escolha: 1/3; -
segunda escolha: 1/2.

Resultado esperado: **1/6 ≈ 0,1667**.
