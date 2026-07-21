import numpy as np
# Simulando 10.000 lancamentos de 2 moedas
np.random.seed(42)

m1 = np.random.choice(['H', 'T'], size=10000)
m2 = np.random.choice(['H', 'T'], size=10000)
# Evento : Pelo menos uma cara (H) -> P = 3/4 = 75%
p_pelo_menos_uma_cara = np.mean((m1 == 'H') | (m2 == 'H'))
print(f"P(Pelo menos 1 Cara) observada : {p_pelo_menos_uma_cara:.2%}")
# Multiplicacao sem reposicao (1/3 * 1/2 = 1/6)
p_yx = (1/3) * (1/2)
print(f"P(Y 1o e X 2o sem reposicao): {p_yx:.4f} (Teorico: 0.1667)")