# Ejemplo de clasificación de spam con weka

1. Abre la base de datos de spam
1. Elimina las últimas tres variables
1. Experimentos con la base de datos original y anotar el rendimiento
1. Aplicar filtro de selección de atributos CfsSubsetEval (Filters->Supervised->Attribute)
1. Aplicar el filtro NumericToBinary
1. Volver a probar experimentos con los métodos.

Si tienes curiosidad, muchos filtros de spam funcionan con un modelo Naïve Bayes. Más información en https://es.wikipedia.org/wiki/Filtrado_bayesiano_de_spam
