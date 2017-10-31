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
    
def load_csv(fichero):
    """Lee un fichero CSV y devuelve la matrix de variables X y el vector de
       etiquetas y
    """
    data = np.loadtxt(fichero, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    return X, y

# Cargamos los datos de entrenamiento
X_train, y_train = load_csv('../datasets-clasificacion/train_countryriskmoodys.csv')

# Cargamos los datos de generalización
X_test, y_test = load_csv('../datasets-clasificacion/test_countryriskmoodys.csv')
 
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


# =================================================================
# REPETIMOS TODO PERO ESTA VEZ PREPROCESANDO LOS DATOS.
# Escalado de datos (ojo, los datos de test también)
print('Rendimiento con datos escalados')
scaler = preprocessing.MinMaxScaler()
X_train_escalado = scaler.fit_transform(X_train)

# Los datos de test se escalan usando SÓLO INFORMACIÓN DE TRAIN
X_test_escalado = scaler.transform(X_test)


regr.fit(X_train_escalado, y_train)
y_test_predict = regr.predict(X_test_escalado)
accuracy = metrics.accuracy_score(y_test,y_test_predict)
conf_matrix = metrics.confusion_matrix(y_test,y_test_predict)

print('Precisión global:\t%0.2f' % (accuracy) )
accuracy_per_class(conf_matrix)

# =================================================================
print('Rendimiento con datos estandarizados')
scaler = preprocessing.StandardScaler(with_mean=False).fit(X_train)
X_train_escalado = scaler.transform(X_train)
X_test_escalado = scaler.transform(X_test)


regr.fit(X_train_escalado, y_train)
y_test_predict = regr.predict(X_test_escalado)
accuracy = metrics.accuracy_score(y_test,y_test_predict)
conf_matrix = metrics.confusion_matrix(y_test,y_test_predict)
print('Precisión global:\t%0.2f' % (accuracy) )
accuracy_per_class(conf_matrix)


# =================================================================
# Probemos a optimizar el coste cd la regresión logística
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
print('Rendimiento con datos estandarizados y optimizando el parámetro de coste')
regr_cv = linear_model.LogisticRegressionCV(solver='liblinear')

regr_cv.fit(X_train_escalado, y_train)

y_test_predict = regr_cv.predict(X_test_escalado)

accuracy = metrics.accuracy_score(y_test,y_test_predict)

conf_matrix = metrics.confusion_matrix(y_test,y_test_predict)

print('Precisión global:\t%0.2f' % (accuracy) )
accuracy_per_class(conf_matrix)
