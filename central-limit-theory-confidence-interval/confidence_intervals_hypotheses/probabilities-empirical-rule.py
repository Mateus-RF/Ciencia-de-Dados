from scipy import stats

# norm . cdf (z) calcula P(Z <= z)
area_1sd = stats.norm.cdf(1) - stats.norm.cdf( -1) # Media + - 1 SD
area_2sd = stats.norm.cdf(2) - stats.norm.cdf( -2) # Media + - 2 SD
area_3sd = stats.norm.cdf(3) - stats.norm.cdf( -3) # Media + - 3 SD

print (f" Area entre Media +-1SD:{ area_1sd :.2%} ( Esperado : 68.27%) ")
print (f" Area entre Media +-2SD:{ area_2sd :.2%} ( Esperado : 95.45%) ")
print (f" Area entre Media +-3SD:{ area_3sd :.2%} ( Esperado : 99.73%) ")