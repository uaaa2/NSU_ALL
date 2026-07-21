

import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터 -> 데이터 전처리

x = np.array([[[1,2,3,4,5,6,7,8,9,10] ,
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]]])

x= x.T
#print(x.shape) #(10, 3, 1)
#exit()

y = np.array([1,2,3,4,5,6,7,8,9,10])   

# *열=컬럼=피쳐=특성=속성=atribute* 

#2. 모델구성 
model= Sequential()
# model.add(Dense(12 , input_dim= 2 )) 
model.add(Dense(12 , input_shape=(3, ))) # (None,3) 이란 뜻. 열은 상관없음
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') 
model.fit(x,y, epochs=100, batch_size=2) 

#4. 평가, 예측
loss = model.evaluate(x,y) 
print("loss= ", loss) 


result= model.predict(np.array([[9,10],[1.4,1.3],[1,0]]).T) # (1,3,2)이기 때문에 .T사용
print("[[9,10],[1.4,1.3],[1,0]]의 예측값: ",result)
















