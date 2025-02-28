
-2º experimento de cerebros realizado:

-100 epocas
-50 Pretrain + 20
-16 imagenes en el dataset + batch size = 4 => 4 batches/epoca => 400 batches
-Tiempo: 2h

-Conclusiones: Mismo resultado que con otras inicilaizaciones y con muchas epocas, empiezo a pensar que la conclusion es la que pensaba antes
pero mas robustamente y es que obviamente verificaremos que el metodo es correcto cuando el resultado del atlas es exactamente el mismo a pesar 
de inicializar con otra imagen, asi que perfecto y por eso la loss converge tan rapido cercano a 0. Basicamente no partimos en este caso de ninguna 
deformacion inicial, sino de una imagen de un cerebro y el atlas final es obviamente otro cerebro, por lo cual la diferencia es cercana a 0, durante 
todo el proceso y por eso la loss es siempre cercana a 0. La cosa sería por ejemplo deformar el cerebro es entonces cuando SI que la loss cambiaria porque se 
busca un cerebro a partir de una deformacion.

Resumen: Partimos de cerebros sin deformar, de pocos, entonces el resultado es un average de un cerebro, cada atlas en cada epoca es un cerebro entonces
la loss es practicamente siempre cercana a 0, cuando cambiaria? con mas imagenes? con enfermos?

INTERP: LINEAR