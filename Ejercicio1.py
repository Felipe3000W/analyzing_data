from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split
iris_dataset = load_iris()
import pandas as pd
import mglearn as mglearn

import matplotlib.pyplot as plt

#Visualización de contenido del DataSet
#print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#print("Data of iris_dataset: \n{}".format(iris_dataset['data']))
#print(iris_dataset['DESCR'][:193] + "\n...")
#print("Target names: {}".format(iris_dataset['target_names']))
#print("Feature names: \n{}".format(iris_dataset['feature_names']))
#print("Type of data: {}".format(type(iris_dataset['data'])))
#print("Type of data: {}".format(iris_dataset['target']))

#Separamos data de entrenamiento y prueba
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

#print(X_train.shape)
#print(X_test.shape)


#Inspect Data  con Visualizaciones
#CREAMOS un dataframe para la data de entrenamiento
iris_dataframe_train =pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

#print(iris_dataframe_train)

# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe_train, c=y_train, figsize=(15, 15), marker='o', 
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

