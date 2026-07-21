import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])
 #선형회귀로써 점차적으로 이해한다(신기하다)
 
#2. 모델구성 # y = wx+b 이때 w,b에 랜덤한 값이 들어간다.
model= Sequential()
model.add(Dense(1, input_dim=1)) #1층(단층) 레이어

#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') 
model.fit(x,y, epochs=2300)

#4. 평가, 예측
result= model.predict(np.array([7]))
print("7의 예측값: ",result)
