from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRFClassifier

#1 데이터
x,y = load_breast_cancer(return_X_y=True)

x_trian, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333 , train_size=0.8,
    stratify=y 
)
kfold = KFold(n_splits=5,shuffle=True,random_state=123)

#2. 모델 
parameters = {
    'learning_rate' : 0.1,
    'max_depth' : 6,
    'n_estimators' : 200,
}

model = XGBRFClassifier(**parameters) ## **로 딕셔너리 호출, *는 리스트 호출

#3 컴파일, 훈련
start_time = time.time()
model.fit(x_trian, y_train) 
end_time = time.time()
print("걸린 시간:",np.round(end_time-start_time,2))

#4. 평가 예측
print('model.score:', model.score(x_test,y_test))   #이거 써야됨

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))

