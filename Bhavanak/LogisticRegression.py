import pandas as pd
import numpy as np
df=pd.read_csv("/storage/emulated/0/Bhavana/admission1")
print(df.head())
print(df.dtypes)
print(df.shape)
print(df.info())
print(df.describe())
print(df.GRE_Score.describe())
mean_g=df.GRE_Score.mean()
df.GRE_Score=df.GRE_Score.replace({0:mean_g})
print(df.TOEFL_Score.describe())
mean_t=df.TOEFL_Score.mean()
df.TOEFL_Score=df.TOEFL_Score.replace({0:mean_t})
print(df.University_Rating.describe())
mean_ur=df.University_Rating.mean()
df.University_Rating=df.University_Rating.replace({0:mean_ur})
print(df.SOP.describe())
mean_s=df.SOP.mean()
df.SOP=df.SOP.replace({0:mean_s})
print(df.LOR.describe())
mean_l=df.LOR.mean()
df.LOR=df.LOR.replace({0:mean_l})
print(df.CGPA.describe())
mean_c=df.CGPA.mean()
df.CGPA=df.CGPA.replace({0:mean_c})
print(df.Research.describe())
mean_r=df.Research.mean()
df.Research=df.Research.replace({0:mean_r})
y=df['Chance_of_Admit']
X=df.drop('Chance_of_Admit',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20,random_state=1)
from sklearn.linear_model import LogisticRegression
regression_model=LogisticRegression()
regression_model.fit(X_train,y_train)
intercept=regression_model.intercept_[0]
print(intercept)
for idx,col_name in enumerate(X_train.columns):
	print("The co-efficient of {} is {}".format(col_name,regression_model.coef_[0][idx]))
y_pred=regression_model.predict(X_test)
print(y_pred)
print(y_test)
#Evaluation metrics
from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_pred,y_test)
print(conf_matrix)
from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_pred,y_test)
print(acc_score)
new_parameters=[[400,300,100,20,10,5,4.5,2]]
Chance_of_Admit_predicted=regression_model.predict(new_parameters)
print(Chance_of_Admit_predicted)

