# -*- coding: utf-8 -*-

import numpy as np

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('exam_1', np.float),
                     ('exam_2', np.float),
                     ('admission', np.int)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]),
                         np.int(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        self.x_names = [a for a in self.dataset.dtype.fields.keys()][:-1]
        self.y_name = [a for a in self.dataset.dtype.fields.keys()][-1:]
        X = self.dataset[self.x_names]
        y = self.dataset[self.y_name]

        permuted_idxs = np.random.permutation(len(X))
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        self.X_train = X[train_idxs]
        self.X_test = X[test_idxs]
        self.y_train = y[train_idxs]
        self.y_test = y[test_idxs]
        
        return self.X_train, self.X_test, self.y_train, self.y_test
        


def train_test_split(x,y,train_pct):
  registros = len(x) # Cantidad de registros
  x = np.random.permutation(x) # Permuta el dataset
  trains = int(np.floor(train_pct*registros)) # Calcula el corte para 70%
  x_train = x[0:trains] #toma el primero 70% de los registros permutados
  x_test = x[trains:] #idem 20
  y_train = y[0:trains]
  y_test = y[trains:] 
  return x_train,x_test,y_train,y_test #devuelve los 3 splits



# Create an artificial dataset
def create_dataset(n_samples):
    '''
    Crea un dataset sintetico con dos centroides.
    Params:
    n_samples <int> Cantidad de samples
    '''
    centroids = np.array([[1,0,0,0],[0,1,0,0]])
    long_centroids = centroids * np.random.randint(low=1,high=10,size=(2,1))
    moved_long_centroids = np.repeat(long_centroids,n_samples/2,axis=0) 
    noise = np.random.normal(size=(n_samples,centroids.shape[1]))
    data = moved_long_centroids + noise
    cluster_ids = np.array([[0],[1]])
    cluster_ids = np.repeat(cluster_ids,n_samples/2,axis=0)
    return data, cluster_ids


#####################################################
### CLEAN DATA SET ##################################

def normalizer(x):
    means = np.mean(x,axis=0) #Calcula las medias por columna
    stds  = np.std(x,axis=0) # Calcula las std dev por columna
    return (x-means)/stds

def desnanizador(x):
    '''
    Desdenanizador recibe un dataset en formato np.array y 
    le quita todas las filas y columnas que tengan nans.
    No es lo más óptimo, pero es para practicar numpy.
    
    Prefiere eliminar primero un registro antes que eliminar una variable
    Podrìan verse variantes.
    '''
    row_filter =  np.logical_not(np.any(np.isnan(x) ,axis=1))
    x = x[row_filter,:]
    column_filter =  np.logical_not(np.any(np.isnan(x) ,axis=0))
    x = x[:,column_filter]
    
    return x

def desnanizador_promedio(x):
    '''
    Desnaniza reemplazando por el promedio de la columna.
    Si hubiera una columna con todos nan, rompe. 
    '''
    columnas_con_nans = np.where(np.any(np.isnan(x),axis=0))[0] # Genera indices de columnas con nan
    for i in columnas_con_nans: #Loop sobre ese índice
        x_i = x[:,[i]] #Toma prestada una columna con nan
        x_i = x_i[np.logical_not(np.isnan(x_i))] # Le quita los nan a esa columna
        media = np.mean(x_i) #Calcula la media de la columna sin los nans
        x[:,[i]] = np.nan_to_num(x[:,[i]],nan=media) #Reemplaza nan en la columna de X original
    return x #Devuelve el dataset limpio
















