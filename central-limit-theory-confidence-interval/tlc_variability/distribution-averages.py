import numpy as np

np.random.seed(42)

populacao_atrasos = np.random.exponential( scale =15 , size =10000) # Assimetrica !

# Extraindo 10.000 amostras de tamanho N =100
amostras = np.random.choice( populacao_atrasos, size =(10000, 100) )
medias = np.mean( amostras, axis =1)
sd_pop = np.std( populacao_atrasos )
sd_teorico = sd_pop / np.sqrt(100)
sd_observado = np.std( medias )
print (f" Media Populacional :{np. mean ( populacao_atrasos ) :.2f}")
print (f" Media das Medias :{np. mean ( medias ) :.2f}")
print (f"SD Teorico ( sigma / sqrt (n)):{ sd_teorico :.2f}")
print (f"SD Observado das Medias :{ sd_observado :.2f}")