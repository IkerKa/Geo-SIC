
-2º experimento de cerebros realizado:

-500 epocas
-150 Pretrain + 350 atlas optimization
-16 imagenes en el dataset + batch size = 8 => 2 batches/epoca => 1000 batches
-Tiempo: 9'5h

-Conclusiones: Como se pueden ver en los experimentos o en el propio GIF, llega un punto en el que el atlas empieza a modificarse muy muy poco. 
Puede seguir siendo importante que se modifique pero pienso que puede haber riesgo de divergencia(?). Entonces por ahora el numero de epocas lo reducire
de forma que sea mas facil trabajar sin esperar tanto y me pondré a buscar cual sería la proporcion ideal de pre-train/optimization.
Por otro lado parece un proceso bastante fiable porque el average de los cerebros que he usado para entrenar es practicamente identico al atlas obtenido.
La cosa es que igual mas que epocas, que tambien, se necesitarían datos de entrenamiento ya que el atlas si que sale un poco borroso y eso es porque creo que no hay
tanta informacion para que salga tan nitido como en el paper.


PENDIENTE: Tengo la loss y el tiempo en whatsapp, ponerlo en esta carpeta!

INTERP: LINEAR