# rom mpl_toolkits.mplot3d import Axes3D
from IPython.core.pylabtools import figsize
from jedi.api.refactoring import inline
from matplotlib.pyplot import subplot, subplots, xticks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE

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

# for col in categorical_features:
#   fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
#   defaulters[col].value_counts().plot(kind="pie", ax= axes[0],subplots=True)
#   sns.countplot(x = col, hue = 'def_pay', ax= axes[1], data = df_cat)
#   plt.show()

# <---- Visualizing the relation between amount of credit and paying ---->
fig , axes = plt.subplots(ncols=2, figsize= (20,8))
# sns.barplot(data=defaulters, x= 'def_pay', ax= axes[0], y='LIMIT_BAL')
# sns.boxplot(data= defaulters,x='def_pay', ax=axes[1],y='LIMIT_BAL')
# plt.show()

# <---- renaming the columns names for better understanding ---->
defaulters.rename(columns={'PAY_1':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL','PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR'},inplace=True)
defaulters.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG','BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
defaulters.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG','PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)

# <---- Visualizing the amount of each age in data ---->
age_counts = defaulters['AGE'].value_counts().reset_index()
age_counts = age_counts.sort_values("AGE")
# defaulters['AGE'].value_counts().plot(kind='pie', ax=axes[0], subplots= True)
# sns.barplot(data=age_counts, x= 'AGE', y='count', ax=axes[1], hue='count', palette='PuOr' )
# axes[1].set_xticklabels(axes[1].get_xticks(), rotation = 90)
# plt.show()

# <---- Visualize relation between age and paying credit ---->
# sns.boxplot(data= defaulters, x='def_pay', y='AGE')
# plt.show()

# <---- Visualizing the relation between amount of bills ---->
bill_amnt_df = defaulters[['BILL_AMT_SEPT',	'BILL_AMT_AUG',	'BILL_AMT_JUL',	'BILL_AMT_JUN',	'BILL_AMT_MAY',	'BILL_AMT_APR']]
# sns.pairplot(data= bill_amnt_df)
# plt.show()

# <---- Visualizing the amount of paying ---->
pay_col = ['PAY_SEPT',	'PAY_AUG',	'PAY_JUL',	'PAY_JUN',	'PAY_MAY',	'PAY_APR']
# for col in pay_col:
#     sns.countplot(data= defaulters, x=col, hue='def_pay')
#     plt.show()

# pay_amnt_df = defaulters[['PAY_AMT_SEPT',	'PAY_AMT_AUG',	'PAY_AMT_JUL',	'PAY_AMT_JUN',	'PAY_AMT_MAY',	'PAY_AMT_APR', 'def_pay']]
# sns.pairplot(data= pay_amnt_df, hue='def_pay')
# plt.show()

# <----  remediate Imbalance using SMOTE(Synthetic Minority Oversampling Technique) ---->
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(defaulters.iloc[:,0:-1], defaulters['def_pay'])
# print('Original dataset shape', len(defaulters))
# print('Resampled dataset shape', len(y_smote))
balanced_df = pd.DataFrame(x_smote, columns= defaulters.columns.drop('def_pay'))
balanced_df['def_pay']= y_smote
# sns.countplot(data= balanced_df, x='def_pay')
# plt.show()

# <---- Implementing Logistic Regression ---->
balanced_df.drop('ID', axis=1, inplace=True)
balanced_df['Payment_Value'] = balanced_df['PAY_SEPT'] + balanced_df['PAY_AUG'] + balanced_df['PAY_JUL'] + balanced_df['PAY_JUN'] + balanced_df['PAY_MAY'] + balanced_df['PAY_APR']
balanced_df['Dues'] = (balanced_df['BILL_AMT_APR']+balanced_df['BILL_AMT_MAY']+balanced_df['BILL_AMT_JUN']+balanced_df['BILL_AMT_JUL']+balanced_df['BILL_AMT_SEPT'])-(balanced_df['PAY_AMT_APR']+balanced_df['PAY_AMT_MAY']+balanced_df['PAY_AMT_JUN']+balanced_df['PAY_AMT_JUL']+balanced_df['PAY_AMT_AUG']+balanced_df['PAY_AMT_SEPT'])
df_x_values = balanced_df.drop(['def_pay','Payment_Value', 'Dues'], axis=1)
df_y_value =balanced_df['def_pay']

std_scaler = StandardScaler() # scale data using a standard scaler
scaled_x_values = std_scaler.fit_transform(df_x_values)

x_train, x_test, y_train, y_test = train_test_split(scaled_x_values, df_y_value, test_size=0.33, random_state=42, stratify=df_y_value)

# <---- Findig the best parameters for model using "GridSearchCV" ---->
model_params = {
    'linear_regression':{
        'model': LogisticRegression(),
        'params':{
            'penalty':['l1','l2'],
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    }
} # a dictionary that saves parameters for each model to be tested using "GridSearchCV"
model_score = pd.DataFrame(columns=["Model Name","Best Score", "Best Params"]) # a dataframe to save score for each model based on parameters
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'],cv=3, scoring='accuracy', n_jobs=-1, verbose=3) # https://www.youtube.com/watch?v=HdlDYng8g9s and https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf.fit(x_train, y_train)
    model_score.loc[len(model_score)]= [model_name, clf.best_score_, clf.best_params_]
print(model_score.head())