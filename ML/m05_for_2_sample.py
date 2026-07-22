#모델을 for문 안에 넣어서 따로 따로 실행안해도 각 모델이 어떤 값이 나오는지 알 수 있음
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier          #regressor = 분류모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
#1. 데이터

data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True)
]

model_list = [
    LinearSVC(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

data_name_list = [
    '아이리스','브래스트랜서','와인'
]
model_name_list = ['LinearSVC', 'LogisticRegression', 'DecisionTreeClassifier', 'RF']


#2. 모델구성
#12번 돌림__
for i, value in enumerate(data_list) :
    x, y = value
    print("=====================================")
    print(data_name_list[i])
    print(x.shape, y.shape)

    for j, value2 in enumerate(model_list) :
        model = value2
        #3. 컴파일, 훈련 
        model.fit(x,y)
        #4. 평가 예측 
        results = model.score(x,y)
        print(model_name_list[j], "model.score:", results)
        

    
