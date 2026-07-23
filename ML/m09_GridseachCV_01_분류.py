from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

#1 데이터
x,y = load_iris(return_X_y=True)

x_trian, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333 , train_size=0.8,
    stratify=y  #y의 클래스의 개수를 균형있게 잘라라
)
print(y_train)
print(y_test)

# kfold = KFold(n_splits=5,shuffle=True,random_state=123)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=123)


#2. 모델 

parameters = [ 
    {'C': [1,10,100,1000], 'kernel': ['linear', 'sigmoid'], "degree" : [3,4,5]},    #24
    {'C':[1,10,100], 'kernel' : ['rbf'], 'gamma': [0.001, 0.0001]}, #6
    {'C': [1,10,100,1000], 'kernel': ['sigmoid'],                   #36
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4,5]}
]#66번 반복 

model = GridSearchCV(SVC(), parameters,cv=kfold, verbose=1,)
#다른 모델을 래핑하는 함수


#3 컴파일, 훈련
start_time = time.time()
model.fit(x_trian, y_train) #66위에서 돌린 후 

print(f"최적의 매개변수 :", model.best_estimator_)
print(f"최적의 파라미터:", model.best_params_)

#4. 평가 예측
print(f"best_scores:", model.best_score_)   #과적합 가능성이 있다. 

print('model.score:', model.score(x_test,y_test))   #이거 써야됨


y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))

end_time = time.time()

print("걸린 시간:",round(end_time-start_time,2))
