from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBClassifier

#1 데이터
x,y = load_iris(return_X_y=True)

x_trian, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333 , train_size=0.8,
    stratify=y  #y의 클래스의 개수를 균형있게 잘라라
)

# 분류라서 StratifiedKFold
# kfold = KFold(n_splits=5,shuffle=True,random_state=123)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=123)


#2. 모델

parameters = [
    {'n_estimators': [100,200,300], 'max_depth': [3,4,6],
     "learning_rate" : [0.1,0.07,0.01]},                                #27
    {'n_estimators':[100,200], 'max_depth' : [4,6,8],
     "learning_rate": [0.2,0.01], 'min_child_weight':[1,3,5]},          #36
    {'n_estimators': [100,300], "learning_rate":[0.1,0.01],
     'subsample':[0.7,1.0], 'colsample_bytree':[0.7,1.0]},              #16
]#총 79개 조합이지만 n_iter개만 랜덤으로 뽑아서 돌린다

model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1,
                           n_iter=20,           #기본값 10, 몇 개 뽑을지
                           random_state=333,
                           n_jobs=-1,
                           )
#다른 모델을 래핑하는 함수


#3 컴파일, 훈련
start_time = time.time()
model.fit(x_trian, y_train)     #20번 돌린 후

print(f"최적의 매개변수 :", model.best_estimator_)
print(f"최적의 파라미터:", model.best_params_)

#4. 평가 예측
print(f"best_scores:", model.best_score_)   #과적합 가능성이 있다.

print('model.score:', model.score(x_test,y_test))   #이거 써야됨


y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))    #분류는 acc

end_time = time.time()

print("걸린 시간:",round(end_time-start_time,2))
