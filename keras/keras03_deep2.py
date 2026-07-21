import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터
x = np.array([1,2,3,4,5,6]) #두개 이상= 리스트
y = np.array([1,2,3,4,5,6])
 #선형회귀로써 점차적으로 이해한다(신기하다)
 
#2. 모델구성 # y = wx+b 이때 w,b에 랜덤한 값이 들어간다.
model= Sequential()
model.add(Dense(12 , input_dim=1)) #보기쉽게 이렇게 간결하게 써도된다
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') #mse = 평균제곱오차
model.fit(x,y, epochs=100)

#4. 평가, 예측
result= model.predict(np.array([7]))
print("7의 예측값: ",result)


