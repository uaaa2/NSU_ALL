
# 아웃풋 레이어를 두개도 가능하다
#단 이렇게 한다면 성능이 떨어질 수 있다
# 때문에 보통은 모델을 하나 더 만든다

import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터 -> 데이터 전처리

x = np.array([[[1,2,3,4,5,6,7,8,9,10] ,
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]]]).T


print(x.shape) #(10, 3, 1)
#exit()

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [9,8,7,6,5,4,3,2,1,0]]).T   
print(y.shape) #(10, 2)
#exit() 
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
model.add(Dense(2))


#3.컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam') 
model.fit(x,y, epochs=100, batch_size=2) 

#4. 평가, 예측
loss = model.evaluate(x,y) 
print("loss= ", loss) 


result= model.predict(np.array([[10,1.3,0]])) # (1,3,2)이기 때문에 .T사용
print("[[9,10],[1.4,1.3],[1,0]]의 예측값: ",result)

#*오버핏(과적합)을 항상 신경써야함
#지금 데이터는 '모든 데이터'를 사용하였기 떄문에 오버핏이 발생할수 있음
#훈련용 데이터와 시험용 데이터는 다른 것이어야 한다

















