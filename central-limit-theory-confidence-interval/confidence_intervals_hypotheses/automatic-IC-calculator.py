import numpy as np
from scipy import stats

def calcular_ic_api(dados, confianca=0.95):
    n_elem = len(dados)
    xbar_elem = np.mean(dados)
    s_elem = np.std(dados, ddof=1)
    se_elem = s_elem / np.sqrt(n_elem)
    z_critico = stats.norm.ppf(1 - (1 - confianca) / 2)
    margem_elem = z_critico * se_elem

    return xbar_elem, (xbar_elem - margem_elem, xbar_elem + margem_elem), margem_elem


np.random.seed(123)

latencias_api = np.random.normal(loc=120, scale=25, size=60)

media_api, (ic_l, ic_h), me_api = calcular_ic_api(latencias_api)

print(f"Média Amostral da API: {media_api:.2f} ms")
print(f"IC 95% do Tempo Médio: ({ic_l:.2f} ms, {ic_h:.2f} ms) | Margem: ±{me_api:.2f} ms")