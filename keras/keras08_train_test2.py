
import tensorflow as tf                                      
print(tf.__version__)

from keras.models import Sequential                            #keras에서 모델을 구성할 때 Sequential 모델을 사용한다.
from keras.layers import Dense                                 #keras에서 Dense 레이어를 사용한다.
import numpy as np                                             # numpy를 np로 줄여서 사용한다. 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])                                            
y = np.array([1,2,3,4,5,6,7,8,9,10])                                                      
 # [실습] 넘파이 리스트의 슬라이싱 => 7:3으로 잘라라!!   

x_train = x[0:7]  # x_train = np.array([1,2,3,4,5,6,7,]) 0은 생략 가능 x[:0]
x_test  = x[7:10] # x_test = np.array([8,9,10]) 7은 생략 가능 x[7:] 10은 생략 가능 x[7:10] = x[7:] 0은 생략 가능 x[:10] = x[:]
 
print(x_train , x_test) 
exit()



#2. 모델구성                                          
model = Sequential()                                           
model.add(Dense(100, input_shape=(1,))) #(None, 1 )            
model.add(Dense(50)) 
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(1))

    

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')        # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값  
model.fit(x_train , y_train , epochs=100, )                                                     


#4. 평가, 예측
loss = model.evaluate(x_test , y_test)
print("loss = ", loss)

result = model.predict(np.array([11]))
print("11의 예측값 : ", result)




