
import tensorflow as tf                                        
print(tf.__version__)         

from keras.models import Sequential                            #keras에서 모델을 구성할 때 Sequential 모델을 사용한다.
from keras.layers import Dense                                 #keras에서 Dense 레이어를 사용한다.
from sklearn.model_selection import train_test_split           #사이킷런에서 train_test_split 함수를 사용하여 데이터를 학습용과 테스트용으로 나눈다.
import numpy as np                                             # numpy를 np로 줄여서 사용한다. 



#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])                                            
y = np.array([1,2,3,4,5,6,7,8,9,10])                                                      

#[실습] train과 test를 섞어서 랜덤하게 7:3을 뽑는다.
#[힌트] 사이킷런

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=42)

#exit()

print(x_train.shape , x_test.shape)    #(7,) (3,)                                                                
print(y_train.shape , y_test.shape)    #(7,) (3,) 

                          

                

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




