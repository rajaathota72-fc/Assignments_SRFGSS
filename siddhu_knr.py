import pandas as pd
import numpy as np
import pandas_profiling as pp
df=pd.read_csv("/Users/SIDDU/Desktop/admission1.csv")
print(df.head())
print(df.tail())
print(df.describe())
print(df.shape)
print(df.info)
profile=pp.ProfileReport(df)
profile.to_file("Report.html")
df=df.drop("Research",axis=1)
print(df.shape)
mean_G=df.GRE_Score.mean()
df.GRE_Score=df.GRE_Score.replace({0:mean_G})
mean_T=df.TOEFL_Score.mean()
df.TOEFL_Score=df.TOEFL_Score.replace({0:mean_T})
mean_U=df.University_Rating.mean()
df.University_Rating=df.University_Rating.replace({0:mean_T})
mean_C=df.CGPA.mean()
df.CGPA=df.CGPA.replace({0:mean_C})
mean_A=df.Chance_of_Admit.mean()
df.Chance_of_Admit=df.Chance_of_Admit.replace({0:mean_A})
print(df.describe())
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X=df.drop('Chance_of_Admit',axis=1)
y=df[['Chance_of_Admit']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
regression_model=LinearRegression()
print(regression_model.fit(X_train,y_train))
intercept=regression_model.intercept_[0]
print(intercept)
for idx,col_name in enumerate(X_train.columns):
	print("The co-efficient for {} is {}".format(col_name,regression_model.coef_[0][idx]))
from sklearn.metrics import mean_squared_error
y_pred=regression_model.predict(X_test)
print(y_pred)
regression_model_mse=mean_squared_error(y_pred,y_test)
print(regression_model_mse)
import math
mae=math.sqrt(regression_model_mse)
print(mae)
accuracy=regression_model.score(X_test,y_test)
print(accuracy)
new_parameters=[[340,120,4.5,5.0,5.0,10.0,2]]
Chance_of_Admit_predicted=regression_model.predict(new_parameters)
print(Chance_of_Admit_predicted)