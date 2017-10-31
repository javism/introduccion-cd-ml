# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:15:12 2017

@author: Javier Sánchez
"""

from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn import metrics
import numpy as np

def accuracy_per_class(conf_matrix):
    for i, row in enumerate(conf_matrix):
        print('Precisión clase %d:\t%0.2f' % (i, row[i] / sum(row)))
    
def leer_csv(fichero):
    """Lee un fichero CSV y devuelve la matrix de variables X y el vector de
       etiquetas y
    """
    data = np.loadtxt(fichero, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    return X, y

# Cargamos los datos de entrenamiento
X_train, y_train = leer_csv('../datasets-clasificacion/train_countryriskmoodys.csv')

# Cargamos los datos de generalización
X_test, y_test = leer_csv('../datasets-clasificacion/test_countryriskmoodys.csv')
 
# Creamos un clasificador de tipo regresión logística
regr = linear_model.LogisticRegression()

print('Rendimiento con datos originales')
# Entrenamos con los datos de pruebas
regr.fit(X_train, y_train)

#Predecimos con datos no vistos
y_test_predict = regr.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_test_predict)
conf_matrix = metrics.confusion_matrix(y_test,y_test_predict)
print('Precisión global:\t%0.2f' % (accuracy) )
accuracy_per_class(conf_matrix)
