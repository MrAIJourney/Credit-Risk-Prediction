# rom mpl_toolkits.mplot3d import Axes3D
from IPython.core.pylabtools import figsize
from jedi.api.refactoring import inline
from matplotlib.pyplot import subplot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('UCI_Credit_Card.csv',delimiter=',')
# print(data.shape)
# print(data.head())
# print(data.describe().T)

# <---- copy data to new dataframe so we can manipulate it ---->
defaulters = data.copy()
defaulters.rename(columns={'default.payment.next.month':'def_pay', 'PAY_0':'PAY_1'}, inplace = True) # rename some columns for better understanding
# print(defaulters.isna().sum()) # check if there are missing data

# <---- Visualizing the data ---->
def_cnt = defaulters['def_pay'].value_counts(normalize=True)*100 # visualizing the Probability Of Defaulting Payment Next Month
# def_cnt.plot.bar()
# plt.title("Probability Of Defaulting Payment Next Month",fontsize= 15)
# for index, value in enumerate(def_cnt):# write the value of each bar on top of it
#     plt.text(index-0.1, value +0.25, str(value), fontsize = 12)
# plt.show()

# <---- Changing Education column so there are only 1 value representing 'other' instead of 4 value '0,4,5,6' ---->
for index, value in enumerate(defaulters['EDUCATION']):
    if (value == 5) | (value == 6) | (value ==0):
        defaulters['EDUCATION'][index]= 4

# print(defaulters['EDUCATION'].value_counts())

# <---- Changing Marriage column so there are only 1 value representing 'other' instead of 4 value '0,3' ---->
for index, value in enumerate(defaulters['MARRIAGE']):
    if value == 0:
        defaulters['MARRIAGE'][index]= 3
# print(defaulters['MARRIAGE'].value_counts())

# <---- Plotting categorical features ---->
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
df_cat = defaulters[categorical_features] # showing each value corresponding with which categorical value
df_cat.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)
df_cat['def_pay']= defaulters['def_pay'] # adding 'def_pay' column at the end of categorical dataframe

for col in categorical_features:
  fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
  defaulters[col].value_counts().plot(kind="pie", ax= axes[0],subplots=True)
  print(axes[0])
  sns.countplot(x = col, hue = 'def_pay', ax= axes[1], data = df_cat)
  plt.show()






