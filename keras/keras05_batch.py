
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터 -> 데이터 전처리
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])



#2. 모델구성 
model= Sequential()
model.add(Dense(12 , input_dim=1)) 
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') #mse = 평균제곱오차
model.fit(x,y, epochs=100, batch_size=2) #만약 데이터가 수없이 많다면? = 그룹으로 잘라서 계산 = batch_size


#4. 평가, 예측
loss = model.evaluate(x,y) #마지막 계산결과에서 나온 y'와 실제값을 mse로 비교해본다
print("loss= ", loss) 


result= model.predict(np.array([7]))
print("7의 예측값: ",result)








