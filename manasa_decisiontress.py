import numpy as np
import pandas as pd
import pandas_profiling as  pp
df=pd.read_csv("/Users/kaasu/OneDrive/Desktop/admission1.csv")
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.info())
#profile=pp.ProfileReport(df)
#profile.to_file("/Users/kaasu/OneDrive/Desktop/report_admission.html")
from sklearn.model_selection import train_test_split
y=df[['Chance_of_Admit']]
X=df.drop('Chance_of_Admit',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
#model building
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)
print(y_test)
#evaluation metrics
from sklearn.metrics import confusion_matrix,accuracy_score
conf_mat=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test,y_pred)
print(conf_mat)
print(acc_score)
#predeployment phase
y_predict_new=model.predict([[401,300,100,2,4.5,3.3,9.3,1]])
print(y_predict_new)