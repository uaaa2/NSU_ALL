#단층 퍼셉트론의 한계를 다중 퍼셉트론 구성으로 해결
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


#1. XOR 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
model = Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 / 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
 
model.fit(x_data,y_data,batch_size=1,epochs=100)


#4. 평가, 예측
y_predict = np.array(model.predict(x_data))                    #훈련한 모델에 x_data를 넣음

