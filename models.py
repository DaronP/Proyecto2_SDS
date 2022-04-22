import re
import time
import gc
#import torch
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import torch.nn as nn
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Importar data
df = pd.read_csv('./preprocess_pt2.csv')

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
    
    return accuracy, cm

def output_results(model, model_name, X_train, y_train, X_score, y_score):
    score, cm = score_model(model, X_train, y_train, X_score, y_score)

    print(f"Predicting with " + model_name + " :\n")
    print(f"{100*score:0.1f}% accuracy\n")
    print("confusion matrix")
    print(cm)

X_train = feature_matrix_train
y_train = target_train
X_score = feature_matrix_test
y_score = target_test

#Modelo 1
model_name = "Logistic Regression"
model = LogisticRegression(solver='lbfgs', max_iter=150)

output_results(model, model_name, X_train, y_train, X_score, y_score)

#Modelo 2
model_name = "K-Nearest Neighbor Classifier"
model = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)

output_results(model, model_name, X_train, y_train, X_score, y_score)