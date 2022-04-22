from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

db = pd.read_csv('./preprocess_pt3.csv')

print(db['HasDetections'].value_counts())