from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.metrics import r2_score
import time
from xgboost import XGBRegressor

#1 데이터
x,y = fetch_california_housing(return_X_y=True)

x_trian, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333 , train_size=0.8,

)
print(y_train)
print(y_test)

# kfold = KFold(n_splits=5,shuffle=True,random_state=123)
kfold = KFold(n_splits=5,shuffle=True,random_state=123)


#2. 모델 

parameters = [ 
    {'n_estimators': [100,150], 'max_depth': [4,8,10], 
     "learning_rate" : [0.1,0.07,0.001]},  #학습률                     #18개
    {'max_depth':[4,8,9,11], 'learning_rate' : [0.2,0.01,0.004],},     #12개
    { 'min_child_weight':[2, 3, 5, 8], 'learning_rate':[0.2,0.01,0.001]}#12개 
]

model = GridSearchCV(XGBRegressor(), parameters,cv=kfold, verbose=1,)


#3 컴파일, 훈련
start_time = time.time()
model.fit(x_trian, y_train) #66위에서 돌린 후 

print(f"최적의 매개변수 :", model.best_estimator_)
print(f"최적의 파라미터:", model.best_params_)

#4. 평가 예측
print(f"best_scores:", model.best_score_)   #과적합 가능성이 있다. 

print('model.score:', model.score(x_test,y_test))   #이거 써야됨


y_predict = model.predict(x_test)
print('accuracy_score :', r2_score(y_test, y_predict))

end_time = time.time()

print("걸린 시간:",np.round(end_time-start_time,2))
