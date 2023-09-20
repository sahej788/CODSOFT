import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from collections import Counter
try:
    train_df=pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")
    test_df=pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv")
except:
    train_df=pd.read_csv("fraudTrain.csv")
    test_df=pd.read_csv("fraudTest.csv")
train_df.head()
fig=px.pie(values=train_df["is_fraud"].value_counts(),names=["Genuine","Fraud"],width=600,height=300,color_discrete_sequence=["orange","black"],title="Fraud vs Genuine transactions")
fig.show()
plt.figure(figsize=(3,4))
ax=sns.countplot(x="is_fraud",data=train_df,palette="pastel")
for i in ax.containers:
    ax.bar_label(i,)
print("Genuine:",round(train_df["is_fraud"].value_counts()[0]/len(train_df)*100,2),"% of the dataset")
print("Frauds:",round(train_df["is_fraud"].value_counts()[1]/len(train_df)*100,2),"% of the dataset")
train_df.info(),test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
drop_columns=["Unnamed: 0","cc_num","merchant","trans_num","unix_time","first","last","street","zip"]
train_df.drop(columns=drop_columns,inplace=True)
test_df.drop(columns=drop_columns,inplace=True)
print(train_df.shape)
print(test_df.shape)
train_df["trans_date_trans_time"]=pd.to_datetime(train_df["trans_date_trans_time"])
train_df["trans_date"]=train_df["trans_date_trans_time"].dt.strftime("%Y-%m-%d")
train_df["trans_date"]=pd.to_datetime(train_df["trans_date"])
train_df["dob"]=pd.to_datetime(train_df["dob"])

test_df["trans_date_trans_time"]=pd.to_datetime(test_df["trans_date_trans_time"])
test_df["trans_date"]=test_df["trans_date_trans_time"].dt.strftime("%Y-%m-%d")
test_df["trans_date"]=pd.to_datetime(test_df["trans_date"])
test_df["dob"]=pd.to_datetime(test_df["dob"])
train_df["age"]=train_df["trans_date"]-train_df["dob"]
train_df["age"]=train_df["age"].astype("timedelta64[Y]")

test_df["age"]=test_df["trans_date"]-test_df["dob"]
test_df["age"]=test_df["age"].astype("timedelta64[Y]")
train_df["trans_month"]=pd.DatetimeIndex(train_df["trans_date"]).month
train_df["trans_year"]=pd.DatetimeIndex(train_df["trans_date"]).year
train_df["latitudnal_distance"]=abs(round(train_df["merch_lat"]-train_df["lat"],3))
train_df["longitudnal_distance"]=abs(round(train_df["merch_long"]-train_df["long"],3))

test_df["latitudinal_distance"]=abs(round(test_df["merch_lat"]-test_df["lat"],3))
test_df["longitudnal_distance"]=abs(round(test_df["merch_long"]-test_df["long"],3))
drop_columns=["trans_date_trans_time","city","lat","long","job","dob","merch_lat","merch_long","trans_date","state"]
train_df.drop(columns=drop_columns,inplace=True)
test_df.drop(columns=drop_columns,inplace=True)
train_df.gender=train_df.gender.apply(lambda x:1 if x=="M" else 0)
test_df.gender=test_df.gender.apply(lambda x:1 if x=="M" else 0)
train_df= pd.get_dummies(train_df, columns=["category"], prefix="category")
test_df= pd.get_dummies(test_df, columns=["category"], prefix="category")

test_df=test_df.reindex(columns=train_df.columns,fill_value=0)
train_df.head()
test_df.head()
X_train=train_df.drop("is_fraud",axis=1)
y_train=train_df["is_fraud"]
X_test=test_df.drop("is_fraud",axis=1)
y_test=test_df["is_fraud"]
from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
clf=DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test,y_pred)
print(report)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
import xgboost as xgb

# Create an XGBoost classifier
clf = xgb.XGBClassifier(
    learning_rate=0.1,  # Learning rate (controls step size during training)
    n_estimators=100,   # Number of boosting rounds (trees)
    max_depth=3,        # Maximum tree depth
    objective='binary:logistic',  # Binary classification problem
    random_state=42
)

# Train the classifier on the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Generate a classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)
