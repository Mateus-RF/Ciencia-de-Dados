from scipy import stats
import numpy as np

n = 50
xbar = 3.2
s = 1.74

# 1. Erro Padrao (SE)
se = s / np.sqrt(n)  # 0.2461

# 2. Valor critico z* para 95% (1.96)
z_95 = stats.norm.ppf(0.975)

# 3. Margem de Erro e Intervalo
margem = z_95 * se  # 0.4823

ic_low, ic_high = xbar - margem, xbar + margem
print(f"SE: {se:.4f} | Margem de Erro: {margem:.4f}")
print(f"Intervalo de Confianca de 95%: ({ic_low:.2f}, {ic_high:.2f})")