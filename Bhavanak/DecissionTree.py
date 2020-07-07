# decision trees
import pandas as pd
import numpy as np
df=pd.read_csv("/storage/emulated/0/Bhavana/admission1")
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.shape)
y=df['Research']
X=df.drop('Research',axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
# evaluation metrics
from sklearn.metrics import accuracy_score,confusion_matrix
conf_mat = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_pred,y_test)
print(conf_mat)
print(acc_score)
new_parameters=[[240,100,9,5,4,3.1,2,1]]
Chance_of_Admit_predicted=model.predict(new_parameters)
print(Chance_of_Admit_predicted)