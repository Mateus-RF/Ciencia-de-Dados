import numpy as np
import pandas as pd

any_numbers = np.array([1, 2, 2, 10])
media = np.mean(any_numbers)  # 3.75
desvios = any_numbers - media  # [-2.75, -1.75, -1.75, 6.25]
desvios_quad = desvios ** 2  # [7.5625, 3.0625, 3.0625, 39.0625]
variancia = np.mean(desvios_quad)  # 13.1875
sd = np.sqrt(variancia)  # 3.6314

df_calculo = pd.DataFrame({
	'x': any_numbers,
	'x - media': desvios,
	'(x - media)^2': desvios_quad,
	'Z-score': (any_numbers - media) / sd,
})

print(df_calculo)
print(f"Resultado Final: Media={media:.2f}, Variancia={variancia:.2f}, SD={sd:.2f}")