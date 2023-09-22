# packages required :

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
      
#Importing libraries required:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the data:
data = pd.read_csv('C:/Users/hp/OneDrive/Desktop/churn/Churn_Modelling.csv')

#Exploring the data:
data.head()
data['Exited'].value_counts().plot(kind='bar')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()
sns.countplot(x='Geography',hue='Exited',data=data)
plt.show()
sns.countplot(x='Gender',hue='Exited',data=data)
plt.show()

#Cleaning the Data:
data.isnull().sum()
data.isna().sum()
data.drop_duplicates()
cols_to_drop=['RowNumber','CustomerId','Surname']
data.drop(cols_to_drop,axis=1,inplace=True)
data.head()

#Encoding Category Data:
data.Gender.value_counts()
data.Gender = pd.Categorical(data.Gender).codes
data.Geography.value_counts()
data.Geography = pd.Categorical(data.Geography).codes

#Exploring Corelations
corr = data.corr()
plt.figure(figsize=(10,15))
sns.heatmap(corr,annot=True)
data.head()

#Splittung data:
X = data.drop('Exited',axis=1)
y = data['Exited']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=40)

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 2. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 3. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
