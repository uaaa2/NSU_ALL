from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

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

#3. 모델 

model = DecisionTreeClassifier()

#3 컴파일, 훈련
#4 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold,n_jobs=-1)

print('ACC', scores,
    '\n cross val scor의 평균:', round(np.mean(scores), 4)
    )
