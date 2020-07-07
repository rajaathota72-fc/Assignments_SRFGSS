import numpy as np
import pandas as pd
#import pandas_profiling as  pp
df=pd.read_csv("Device storage/documents/admission1.csv")
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.describe())
print(df.shape)
#profile=pp.ProfileReport(df)
#profile.to_file("Device storage/documents/admission.html")
from sklearn.model_selection import train_test_split
#data wrangling
y=df[['Chance_of_Admit']]
X=df.drop('Chance_of_Admit',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
#model building
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
intercept=model.intercept_[0]
print(intercept)
for idx,col_name in enumerate(X_train.columns):
    print("the cofficient for {} is {}".format(col_name,model.coef_[0][idx]))
#evaluation metrics
y_pred=model.predict(X_test)
print(y_pred)
print(y_test)
from sklearn.metrics import confusion_matrix,accuracy_score
conf_mat=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test,y_pred)
print(conf_mat)
print(acc_score)
#pre-deployment phase
y_pred_new=model.predict([[401,303,104,3,4.6,3.5,9.8,1]])
print(y_pred_new)