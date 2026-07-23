from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#1 데이터
x,y = load_iris(return_X_y=True)

kfold = KFold(n_splits=5,shuffle=True,random_state=123)#n_splits: 데이터를 몇번 나눌건지 
model = DecisionTreeClassifier()

#3 컴파일, 훈련
#4 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold,n_jobs=-1)

print('ACC', scores,
    '\n cross val scor의 평균:', round(np.mean(scores), 4)
    )
