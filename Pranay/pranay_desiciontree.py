import numpy as np
import pandas as pd
import pandas_profiing as np
df = pd.read_csv("C:/Users/USER/Downloads/admission1.csv")
print(df.head())
print(df.dtypes)
print(df.shape)
print(df.info())
print(df.describe())
profile = pp.ProfileReport(df)
profile.to_file("/Users/USER/pycharmprojects/reportDT.html")
y = df['Chance_of_Admit']
x = df.drop('Chance_of_Admit',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
#model building
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
#evoluation matrix
from sklearn.metrics import accuracy_score,confusion_matrix
conf_mat = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_pred,y_test)
print(conf_mat)
print(acc_score)
#predeployment phase
y_predict_new=model.predict([[401,300,100,2,4.5,3.3,9.3,1]])
print(y_predict_new)
