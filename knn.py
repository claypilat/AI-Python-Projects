import sklearn
from sklearn.utils import  shuffle
from sklearn.neighbors import  KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import  linear_model, preprocessing


data = data = pd.read_csv("C:/Users/cpilat/Desktop/Projects/Python/whatever/car.data", sep=',')

print(data.head())