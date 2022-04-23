import re
import time
import gc
#import torch
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
#import torch.nn as nn
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

#Importar data

df = pd.read_csv('./preprocess_pt3.csv')

print(len(df.columns))

#Separar train y test
target = df['HasDetections']
feature_matrix = df.drop(['HasDetections'], axis=1)


#print('Final features:', feature_matrix.columns)
feature_matrix.head()

#División de datos de entrenamientos y prueba
feature_matrix_train, feature_matrix_test, target_train, target_test = model_selection.train_test_split(feature_matrix, target, test_size=0.30, random_state=31)


#Métodos para aplicar los modelos
def score_model(model, train_data, train_labels, score_data, score_labels):
    model.fit(X=train_data, y=train_labels)

    # predict and score on the scoring set
    pred = model.predict(score_data)
    
    cm = confusion_matrix(score_labels, pred)
    accuracy = metrics.accuracy_score(y_true=score_labels, y_pred=pred)
    cr = metrics.classification_report(score_labels, pred, target_names=['detected', 'not detected'])
    
    return accuracy, cm, cr

def output_results(model, model_name, X_train, y_train, X_score, y_score):
    print('\nTraining model ', model_name, '... ')
    score, cm, cr = score_model(model, X_train, y_train, X_score, y_score)
    print('Done.\n')

    print(f"Predicting with " + model_name + " :\n")
    print(f"{100*score:0.1f}% accuracy\n")
    print("confusion matrix")
    print(cm)
    print("classification report")
    print(cr)

X_train = feature_matrix_train
y_train = target_train
X_score = feature_matrix_test
y_score = target_test

n_components = 50
train_index = range(0, len(X_train))
score_index = range(0, len(X_score))

#Scaler y Principal Component Analysis
scaler = StandardScaler()
pca = PCA(n_components=n_components, whiten=False, random_state=32)

X_train_st = scaler.fit_transform(X_train)
X_score_st = scaler.fit_transform(X_score)

X_train_pca = pca.fit_transform(X_train_st)
print("Varianza de los 50 componentes: ", sum(pca.explained_variance_ratio_))
X_score_pca = pca.fit_transform(X_score_st)

X_train_pca = pd.DataFrame(data=X_train_pca, index=train_index)
X_score_pca = pd.DataFrame(data=X_score_pca, index=score_index)

X_train_inverse = pca.inverse_transform(X_train_pca)
X_score_inverse = pca.inverse_transform(X_score_pca)

print('Attempting to train all 3 models...')

###Modelo 1
model_name = "Logistic Regression"
model = LogisticRegression(solver='lbfgs', max_iter=150)

model1_strt = time.time()
output_results(model, model_name, X_train_pca, y_train, X_score_pca, y_score)
model1_end = time.time()
print('Elapsed time: ', (model1_end - model1_strt))

###Modelo 2
model_name = "Desicion Tree Classifier"
model = DecisionTreeClassifier()

model2_strt = time.time()
output_results(model, model_name, X_train_pca, y_train, X_score_pca, y_score)
model2_end = time.time()
print('Elapsed time: ', (model2_end - model2_strt))

###Modelo 3
model_name = "Naive Bayes"
model = GaussianNB()

model3_strt = time.time()
output_results(model, model_name, X_train_pca, y_train, X_score_pca, y_score)
model3_end = time.time()
print('Elapsed time: ', (model3_end - model3_strt))