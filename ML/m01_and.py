import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 0, 0, 1])

#2. 모델
model = LinearSVC()

#3. 훈련 
model.fit(x_data,y_data)
                                                     #역전파 계산이 필요 없기 때문에 epochs가 필요없음

#4. 평가, 예측
y_predict = model.predict(x_data)                    #훈련한 모델에 x_data를 넣음

result = model.score(x_data, y_data)    
print(result)                                        #1.0 accuracy지수

acc = accuracy_score(y_data, y_predict)
print(f"acc= {acc}")  