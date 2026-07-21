import pandas as pd
import numpy as np

populacao = np.array(['Aluno_A', 'Aluno_B', 'Aluno_C', 'Aluno_D', 'Aluno_E'])

# Sorteio COM reposicao
print('COM reposicao:', np.random.choice(populacao, size=3, replace=True))

# Sorteio SEM reposicao
print('SEM reposicao:', np.random.choice(populacao, size=3, replace=False))

# Amostragem direta de um DataFrame no Pandas
df_users = pd.DataFrame({'id': range(100), 'ativo': np.random.choice([0, 1], 100)})
amostra = df_users.sample(n=5, random_state=42)
print('Amostra DataFrame:\n', amostra)