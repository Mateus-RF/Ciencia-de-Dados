from scipy import stats

# Example estimates (replace with your sample mean and standard error)
xbar = 0.0
se = 1.0

confs = [0.90, 0.95, 0.98, 0.99]
for c in confs:
	z = stats.norm.ppf(1 - (1 - c) / 2)
	me = z * se
	print(f"Confianca: {c:.0%} | z*: {z:.3f} | Margem: {me:.3f} | IC: ({xbar-me:.2f}, {xbar+me:.2f})")