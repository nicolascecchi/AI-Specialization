# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from exam_01_preprocessing import *
from exam_02_modelling import *
from exam_03_evaluation import *
#Parte 2
#2.1 Datos descargados

#2.2 Cargar el dataset con numpy

dataset = np.genfromtxt('clase_8_dataset.csv',delimiter=',')


#2.3 Graficar scatter
plt.scatter(dataset[:,0],dataset[:,1])
plt.title('Data set original')
plt.show()

#2.4 Train test split en 80-20
  
x = dataset[:,0]
y = dataset[:,1]
xnorm= normalizer(x)
xtrain,xtest,ytrain,ytest = train_test_split(xnorm,y,0.8)
##################################################
# # Parte 3   -   POLINOMIO
##################################################

#Hacer una polinomica
# Parte 3.a
#Con N = 1
# K-Folds implementation
# Criterion = Mean Squared Error (default parameter)
error_list, param_list = k_folds_model(xtrain,ytrain,LinearRegressionWithB())
# Tomo el indice donde se produjo el minimo MSE
index_minimo_error = np.argmin(error_list)
# Retrieves best parameters from parameter list
best_model = param_list[index_minimo_error]

# Parte 3.b
best_models_params = []
best_models = []
for i in range(2,5):
    polinomio = polyfit(degree=i)
    error_list, param_list = k_folds_model(xtrain,ytrain,polinomio)
    # Tomo el indice donde se produjo el minimo MSE
    index_minimo_error = np.argmin(error_list)
    # Retrieves best parameters from parameter list
    best_model = param_list[index_minimo_error]
    best_models_params.append(np.array(best_model))
    print('for degree {} the best params are {}'.format(i,best_model))

#for degree 2 the best params are
#array([[ 33.13094908],
#       [-22.4471002 ],
#       [ 16.85062331]])
#for degree 3 the best params are 
#array([[-12.95454814],
#       [ 33.19692007],
#       [ -0.38735833],
#       [ 17.99520712]])
#for degree 4 the best params are 
#array([[-3.20395010e-02],
#       [-1.29526745e+01],
#       [ 3.32758263e+01],
#       [-3.89889222e-01],
#       [ 1.79710369e+01]])

dict_errores = {}
mse = MSE()

# Trabajo con el polinomio de grado 2
W_2 = best_models_params[0]

xtrain_2 = X_expander(xtrain,2)
train_prediction_2 = xtrain_2 @ W_2

dict_errores.update({'degree2_train':mse(target=train_prediction_2,prediction=ytrain)})


xtest_2 = X_expander(xtest,2)
test_prediction_2 = xtest_2 @ W_2
dict_errores.update({'degree2_test':mse(target=test_prediction_2,prediction=ytest)})


#W_3 = best_models_params[1]
W_3 = best_models_params[1]

xtrain_3 = X_expander(xtrain,3)
train_prediction_3 = xtrain_3 @ W_3

dict_errores.update({'degree3_train':mse(target=train_prediction_3,prediction=ytrain)})


xtest_3 = X_expander(xtest,3)
test_prediction_3 = xtest_3 @ W_3
dict_errores.update({'degree3_test':mse(target=test_prediction_3,prediction=ytest)})



#W_4 = best_models_params[2]
W_4 = best_models_params[2]

xtrain_4 = X_expander(xtrain,4)
train_prediction_4 = xtrain_4 @ W_4

dict_errores.update({'degree4_train':mse(target=train_prediction_4,prediction=ytrain)})


xtest_4 = X_expander(xtest,4)
test_prediction_4 = xtest_4 @ W_4
dict_errores.update({'degree4_test':mse(target=test_prediction_4,prediction=ytest)})


# Diccionario final de errores
#{'degree2_train': 918462.5209077444,
# 'degree2_test': 258869.27345862455,
# 'degree3_train': 951452.9365189056,
# 'degree3_test': 269528.89615404047,
# 'degree4_train': 951511.5607813575,
# 'degree4_test': 269503.3395450088}

# Vemos que los errores son todos muy similares en cuanto a dimensiones,
# Matematicamente daba mejor usar degree2_test aca, pero la forma de los datos
# ya nos damos cuenta que no es una parabola,
# por lo tanto tomo el siguiente nivel, grado 3 que tiene mejor pinta el polinomio
# al ver que tiene 2 puntos de inflexion claros en los graficos

# Punto 3.C
# Se selecciona el modelo que tenga menor TEST error
# En este caso particular es degree3_test el menor
# Me resulta sospechoso que el TEST error sea menor que el TRAIN, 
overall_best = min(dict_errores, key=dict_errores.get)

# Graficos
plt.scatter(xtrain, train_prediction_3,label='Training',color='red')
plt.scatter(xtest,test_prediction_3,label='Test',color='blue')
plt.scatter(xtest,ytest,label='original',color='orange')
plt.title('Original y prediccion sobre X normalizada')
plt.legend()
plt.show()


######################################################
###PARTE 4 -- Minibatch Gradient Descent 
######################################################

#4.a Para cada epoch calcular error de train y validacion

# Tras haber intentado con el dataset original, me tiraba todo NaNs
# Pude determinar que se tratabade un tema numerico, me explotaba el gradiente
# Por lo que primero paso a normalizar los datos
# Lo hago separado porque entiendo que es buena practica no "contaminar" con las medias
# entree train y test
#x_train_norm = normalizer(xtrain)
#xtrain_3 = X_expander(x_train_norm,3)

#x_test_norm = normalizer(xtest)
#xtest_3 = X_expander(x_test_norm,3)


W_mini_batch, train_error_log, validation_error_log = mini_batch_gradient_descent(X_train=xtrain_3,
                                                                       Y_train=ytrain.reshape(-1,1),
                                                                       b=16,
                                                                       learning_rate=0.001,
                                                                       epochs=10000)
prediction_mini_batch =  xtest_3 @ W_mini_batch 

# Parte 4.B
# Grafico de los errores
# Se encuentran las salidas en la carpeta

plt.scatter(xtest,prediction_mini_batch)
plt.title('Prediccion de MiniBatch sobre xtest noramizado')
plt.show()


f, ax = plt.subplots(figsize=(10,10),ncols=2,nrows=2)
ax1,ax2,ax3,ax4 = ax.flatten()
ax1.plot(train_error_log)
ax1.set_title('Train error log')
ax2.plot(train_error_log[:75])
ax2.set_title('Train error log primeras 75 epochs')
ax3.plot(validation_error_log)
ax3.set_title('Validation error log')
ax4.plot(validation_error_log[:75])
ax4.set_title('Train error log primeras 75 epochs')

plt.show()

f, ax = plt.subplots()
ax.plot(train_error_log[:75],color='green',label='Train')
ax.plot(validation_error_log[:75],color='orange',label='Validation')
ax.set_title('Validation vs Train error evolution over first 75 epochs')
ax.legend()
plt.show()

#Parte 4.c
# Analisis de resultados
#
# W_minibatch
#array([[-12.8975913 ],
#       [ 33.09440005],
#       [ -0.50365608],
#       [ 17.99626227]])
# W_3
#array([[-12.95454814],
#       [ 33.19692007],
#       [ -0.38735833],
#       [ 17.99520712]])
# Vemos que los pesos que consigue minibatch y el modelo polinomico son extremadamente parecidos, 
# tambien comparando los graficos, vemos que los resultados son muy similares.
# De los graficos de los errores en las epochs, vemos que las 50k fueron exageradas y que 
# ya a partir de epoch=75 el error ya es muy chiquito.
#
#

