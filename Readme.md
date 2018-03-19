Este software calcula la matriz de co-ocurrrencia y extrae los 13 descriptores de Haralick, los imprime en consola y los añade al archivo “features_vectors.csv” ubicado en la carpeta out.

Para la compilación de dicho software se ejecuta las siguientes lineas en consola:
	$ cmake .
	$ make

Para su ejecución:
 	$ ./main image.png angulo y distancia

Donde el angulo puede ser 0, 45, 90, 135 o 180, y la distancia debe ser mayor a 0.
Ejemplo:
 
	$ ./main image.png 180 1
Esto calcularía los descriptores de Haralick con una matriz de co-ocurrencia en 180° a una distancia de un pixel.

	

Reference pages:

http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
http://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html
 
