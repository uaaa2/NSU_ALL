from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#1 데이터
x,y = fetch_california_housing(return_X_y=True)

kfold = KFold(n_splits=5,shuffle=True,random_state=123)
model = RandomForestRegressor(max_depth=5, max_leaf_nodes=3, n_estimators=100,
                              )

#3 컴파일, 훈련
#4 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold,n_jobs=-1)

print('ACC', scores,
    '\n cross val scor의 평균:', round(np.mean(scores), 4)
    )
