#대체로 input_shape 를 사용. x가 여러 차원일수 있기떄문 (ex) input_shape=(28,28,1)



import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #행렬연산에 매우 특화

#1. 데이터 -> 데이터 전처리

x = np.array([[1,2,3,4,5,6] , [7,8,9,10,11,12]]) # (2,6)    
# x = x.T #transpose
x = x.transpose() #데이터 바꾸는 노가다 하기싫으니까 함수로 바꾸기
print(x.shape)
#exit()

y = np.array([1,2,3,4,5,6]) # (6, )  

# *열=컬럼=피쳐=특성=속성=atribute* 

#2. 모델구성 
model= Sequential()
# model.add(Dense(12 , input_dim= 2 )) 
model.add(Dense(12 , input_shape=(2, ))) 
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


result= model.predict(np.array([[7,13]])) # (1,2) 가 나올거임
print("7의 예측값: ",result)














