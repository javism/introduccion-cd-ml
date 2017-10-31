# Instrucciones para ejecutar un ejemplo básico que utilice Weka como biblioteca de Machine Learning

Este ejemplo entrena diferentes modelos de aprendizaje automático y prueba su rendimiento con un conjunto de datos de generalización (no utilizados durante el entrenamiento). Más instrucciones dentro del código fuente Java. Puedes descargar Weka de https://www.cs.waikato.ac.nz/ml/weka/

Compilar (con la ruta a `weka.jar` adecuada):
```
javac -cp ~/programas/weka-3-8-1/weka.jar ClasificadorWeka.java
```

Ejecutar (con la ruta a `weka.jar` adecuada):
```
java -cp ~/programas/weka-3-8-1/weka.jar:.  ClasificadorWeka
```
