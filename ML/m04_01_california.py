import numpy as np
from sklearn.datasets import fetch_california_housing

x,y = fetch_california_housing(return_X_y=True)
print(x.shape,y.shape)

#2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor #regressor = 분류모델
from sklearn.ensemble import RandomForestRegressor

#model = LinearSVC()                분류라 안됨
#model = LogisticRegression()       분류라 안됨
#model = DecisionTreeRegressor()    #1.0
model = RandomForestRegressor()     #0.9742923584150744



#decisiontree 
#3. 컴파일, 훈련
model.fit(x,y)

#4. 평가, 예측
result = model.score(x,y)
print(result)
