import numpy as np

# Simulando 10.000 experimentos de 30 moedas sob H0 (p = 0.5)
np.random.seed(42)
simulacoes = np.random.binomial(n=30, p=0.5, size=10000)
ic_low = np.percentile(simulacoes, 2.5)   # ~10
ic_high = np.percentile(simulacoes, 97.5) # ~20
p_valor = np.mean(simulacoes >= 22)      # ~0.0069 (0.69%)

print(f"Intervalo Empírico de 95% sob H0: [{ic_low:.0f}, {ic_high:.0f}] caras")
print(f"P-Valor (Probabilidade de N >= 22): {p_valor:.4f}")
if p_valor < 0.05:
	print("Rejeitamos H0! A moeda possui viés estatisticamente significante.")