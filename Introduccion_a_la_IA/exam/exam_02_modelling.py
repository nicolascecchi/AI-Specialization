import numpy as np


class BaseModel(object):
  def __init__(self):
    self.model = None

  def fit(self,x,y):
    #train del model
    return NotImplemented

  def predict(self,x):
    #returns yhat
    return NotImplemented


class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1: #Si X es unidimensional
            W = (X.T @ y) / (X.T @ X) # Numero, la inversa pasa a ser una division
        else: #Si X  tiene N>1 dimensiones (ejemplo, polinomios) 
            W = np.linalg.inv(X.T @ X) @ X.T @ y # N x 1
        self.model = W # n x 1

    def predict(self, X):
        return X * self.model # Dimensionssolved w/ Broadcasting


class AverageBaseModel(BaseModel):
    def fit(self,y):
        self.model = y.mean()
    def predict(self,x):
        prediction = self.model * np.ones(shape=(len(x),1))
        return prediction


class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        y = y.reshape((-1,1))
        X = X.reshape((-1,1)) # mx1
        X_expanded = np.hstack((X, np.ones(X.shape))) # mx2
        W = np.linalg.inv(X_expanded.T @ X_expanded) @ X_expanded.T @ y
        self.model = W # 2x1

    def predict(self, X):
        X = X.reshape((-1,1))
        X_expanded = np.hstack((X, np.ones(X.shape)))
        return (X_expanded @ self.model).T





class polyfit(BaseModel):
    
    def _poly_generator_(self,X):
        # Arma la matriz de X^j
        ones = np.ones(shape=(X.shape[0],1))
        X_poly = ones.copy()
        for k in range(1,self.degree+1):
            X_poly= np.hstack((ones*(X**k) , X_poly))
        return X_poly
        
    def fit(self,X,y,degree):
        #rows = X.shape[0]
        #features = X.shape[1]
        X = X.reshape((-1,1)) # M x 1
        y = y.reshape((-1,1)) # M x 1
        self.degree = degree # Grado
        self.train_x = self._poly_generator_(X)        
        self. linearmodel = LinearRegression()
        self.linearmodel.fit(self.train_x,y)
        self.model = self.linearmodel.model
        
    def predict(self,X):
        X = X.reshape((-1,1))
        self.predict_X = self._poly_generator_(X)
        return self.predict_X @ self.model
        



###################################################################
######## CLASSIFICATION ###########################################
###################################################################
class Kmeans(BaseModel):

    def redefine_centroids(self, X, centroids, n_clusters):
        distance = np.sqrt(np.sum((centroids[: ,None] - X )**2, axis=2))
        centroid_with_min_distance = np.argmin(distance, axis=0)
        for i in range(centroids.shape[0]):
            centroids[i] = np.mean( X[centroid_with_min_distance == i, :], axis = 0)
        return centroids, centroid_with_min_distance

    def fit(self, X, n_clusters,MAX_ITER=20):
        centroids = np.eye(n_clusters, X.shape[1] ) *10 + np.random.random(size=[n_clusters, X.shape[1]] ) *2
        for i in range(MAX_ITER):
            centroids, clusters = self.redefine_centroids(X, centroids, n_clusters)
        self.model = centroids
        return  NotImplemented

    def predict(self, X):

        centroids = self.model
        distance = np.sqrt(np.sum((centroids[:, None] - X)** 2, axis=2))
        centroid_with_min_distance = np.argmin(distance, axis=0)
        return centroid_with_min_distance



class LogisticRegression(BaseModel):
  
    def _sigmoid_(self,W,X):
        exponent =  X @ W
        return 1/(1+np.exp(-exponent))
      
    def _decision_(self,score,boundary):
        
        return (score >= self.model_boundary) * 1
    
    def fit(self, X, Y, model_boundary = 0.5,epochs=5000,lr=0.01,gdtype='batch'):        
        self.model_boundary = model_boundary
        self.epochs = epochs
        self.lr=lr
        self.Y = Y.reshape((-1,1))
        n = X.shape[0]
        m = X.shape[1]
        
        self.W = np.random.randn(m).reshape((m,1))
        
        if gdtype == 'batch':
            for e in range(self.epochs):
                prediction = self._sigmoid_(self.W,X)
                error = (prediction.reshape((-1,1)) - self.Y)
                grad_sum = np.sum(error * X)
                grad_mul = 1/n * grad_sum
                gradient = grad_mul.T.reshape(-1,1)
                self.W = self.W - (self.lr * gradient)

        elif gdtype == 'mini_batch':
            b=16
            for i in range(self.epochs):
                idx = np.random.permutation(X.shape[0])
                X = X[idx]
                Y = Y[idx]
                batch_size = int(len(X) / b)
                
                for i in range(0, len(X), batch_size):
                    end = i + batch_size if i + batch_size <= len(X) else len(X)
                    batch_X = X[i: end]
                    batch_y = Y[i: end]
                    exponent = np.sum(np.transpose(self.W) * batch_X, axis=1)
                    prediction = 1/(1 + np.exp(-exponent))
                    error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)
        
                    grad_sum = np.sum(error * batch_X, axis=0)
                    grad_mul = 1/b * grad_sum
                    gradient = np.transpose(grad_mul).reshape(-1, 1)
        
                    self.W = self.W - (lr * gradient)
                    
        self.model = self.W

    def predict(self,X):
        proba_prediction = self._sigmoid_(self.model,X)
        class_prediction = self._decision_(proba_prediction,self.model_boundary)
        return class_prediction
    
    
    
    




###############################################################
############### GRADIENT DESCENTS #############################
###############################################################

#### MSE GRADIENT DESCENT

def gradient_descent(X_train,Y_train, learning_rate=0.01, epochs=5000):
  n = X_train.shape[0] # Cantidad de muestras
  m = X_train.shape[1] # Cantidad de features
  
  # Inicializa pesos aleatorios
  W = np.random.randn(m).reshape(m,1) #Inicializa vector aleatorio
                                      #Dimensión de las features
                                      #Lo dimensiona en formato columna
  for i in range(epochs): # Para cada epoch recorre todo el dataset
    prediction = X_train @ W # Predice con los pesos actuales
    error = Y_train - prediction #Calcula el error

    grad_sum = np.sum(error * X_train,axis=0) #Multiplica el error por los X
                                        # Suma sobre cada feature (por columna)
    grad_mul = -2/n * grad_sum
    gradient = np.transpose(grad_mul).reshape(-1,1) #mx1

    W = W - (learning_rate * gradient) # Update de los weights
  return W



def stochastic_gradient_descent(X_train,Y_train, learning_rate=0.01,epochs=5000):
  n = X_train.shape[0] #nb of samples
  m = X_train.shape[1] #nb of features

  W = np.random.randn(m).reshape(m,1) #Rand init of weights

  for i in range(epochs):
    idx = np.random.permutation(X_train.shape[0]) #Genera índices aleatorios
    X_train_epoch = X_train[idx]
    Y_train_epoch = Y_train[idx]
    for j in range(n): # Para cada observación
      prediction = np.matmul(X_train_epoch[j].reshape(1,-1),W) #1x1
      error = Y_train_epoch[j] - prediction

      grad_sum = error * X_train[j]
      grad_mul = -2 * grad_sum
      gradient = np.transpose(grad_mul).reshape(-1,1)

      W = W - (learning_rate * gradient)

    return W


def mini_batch_gradient_descent(X_train,Y_train,b=16,learning_rate=0.01,epochs=5000):
    #b = 16 # Number of batches to process
    n = X_train.shape[0] # N samples
    m = X_train.shape[1] # M features
    
    W = np.random.randn(m).reshape(m,1) # Mx1
    
    for i in range(epochs):
        idx = np.random.permutation(n)
        X_train_epoch = X_train[idx]
        Y_train_epoch = Y_train[idx]
      
        batch_size = int(len(X_train)/b)
        
        for j in range(0,len(X_train_epoch),batch_size):
            end = j + batch_size if j + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train_epoch[j:end]
            batch_Y = Y_train_epoch[j:end]
            
            prediction = np.matmul(batch_X,W)
            error = batch_Y - prediction
            
            grad_sum = np.sum(error * batch_X,axis=0)
            grad_mul = -2/batch_size * grad_sum
            gradient = np.transpose(grad_mul).reshape(-1,1)
    
            W = W - (learning_rate * gradient)
    
    return W 




