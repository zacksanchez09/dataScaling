import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#leer el archivo
dataFilePath = "dataSets/breastCancer.csv"

dataFrame = pd.read_csv(dataFilePath)
#revisamos el tipo de dato , para que vean que es un objeto de tipo datafreme de pandas
print(type(dataFrame))
#un vistazo a los datos con los que trabajaremos
print( dataFrame.head())
#revisamos que no haya nulos

print(dataFrame.isnull().sum())
print(dataFrame.describe ())
print(dataFrame.radius_mean.describe())
#Ploteamos la cantidad de personas con problemas del corazon (1) y los que no tienen (0)
ax = dataFrame['diagnosis'].value_counts().plot(kind='bar');

#preparar la entrada y la posible salida
dataFrame.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)

#creamos los escaladores
scalerMM = MinMaxScaler()
scalerS = StandardScaler()

y = dataFrame['diagnosis'].values
x = dataFrame.drop(["diagnosis", "Unnamed: 32", "id"], axis = 1)
x.hist(column=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se'])

scaledXDFMM = x.columnpy()
scaledXDFS = x.copy()

#escalar los valores de entradas
scaledXMM = scalerMM.fit_transform(x)
scaledXDFMM[x.columns ] = scalerMM.fit_transform(x)
scaledXDFMM.hist(column=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se'])

scaledXS = scalerS.fit_transform(x)
scaledXDFS[x.columns ] = scalerS.fit_transform(x)
scaledXDFS.hist(column=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se'])

#SPlitting into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3)
#creamos el modelo de un Perceptron multicapa
model = MLPClassifier(alpha=1, max_iter=1000)
#a entrenar
model.fit(xtrain , ytrain)
# Aplicar metrica al modelo

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))
#sacar la prediccion en la parte del test
ytestpred = model.predict(xtest)

#sacar el reporte de clasificacion
print('Classification report: \n', classification_report (ytest, ytestpred))
class_names = [0, 1]
disp = plot_confusion_matrix(model ,xtest , ytest , display_labels = class_names, cmap=plt.cm.Blues ,)
disp.ax.set_title ("Confusion matrix , without normalization")
plt.show()
confusion_matrix (ytest, ytestpred)
