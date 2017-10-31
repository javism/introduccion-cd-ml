# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:06:36 2017

@author: Javier Sánchez
"""

import matplotlib.pyplot as plt
import numpy as np

# Variable independiente
x = [1,2,3,4,4.5, 5, 6, 7, 8]
# Variable dependiente
y = [3,5,7,10,9, 10, 9, 8, 7]

# Ajusta un polinominio de n grados. Si n=1 es una regresión lineal
fit = np.polyfit(x,y,1)
# Creamos un objeto 'poly1d' que nos permite trabajar con el modelo
fit_fn = np.poly1d(fit)
y_predicha = fit_fn(x)

fig, ax = plt.subplots(1,1)
ax.plot(x,y,'yo')
ax.plot(x,y_predicha, 'k--')

# Ajustamos un modelo de grado 2
fit = np.polyfit(x,y,2)
fit_fn = np.poly1d(fit)
y_predicha = fit_fn(x)
ax.plot(x,y_predicha, 'r--')

plt.show()
