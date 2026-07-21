#MLP: mochine layer percentron

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터 -> 데이터 전처리
# x = np.array([[1,2,3,4,5,6] , [7,8,9,10,11,12]]) # (2,6) #2행 6열. 잘못 설계한 데이터
#Q. 위와같은 데이터에 input_dim=6 으로 해놓으면 안되나?
#A. dim 에 상관없이 y가 1이기 때문에 애초에 맞지 않음

x = np.array([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]]) #(6,2) 일단 열의 갯수만 맞추면 돌아간다


y = np.array([1,2,3,4,5,6]) # (6, )  

# *열=컬럼=피쳐=특성=속성=atribute* 

#2. 모델구성 
model= Sequential()
# model.add(Dense(12 , input_dim= 2 )) 
model.add(Dense(12 , input_shape=(2, ))) #벡터가 2개=열 2개란 뜻 이렇게 써도 된다
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') 
model.fit(x,y, epochs=100, batch_size=2) #만약 데이터가 수없이 많다면? = 그룹으로 잘라서 계산 = batch_size


#4. 평가, 예측
loss = model.evaluate(x,y) 
print("loss= ", loss) 


result= model.predict(np.array([[7,13]])) # (1,2) 가 나올거임
print("7의 예측값: ",result)











