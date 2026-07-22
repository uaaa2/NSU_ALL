import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense


#1. XOR 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
model = Sequential()
model.add(Dense(1,input_dim=2, activation='sigmoid'))


#3. 컴파일 / 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
 
model.fit(x_data,y_data,batch_size=1,epochs=100)


#4. 평가, 예측
y_predict = np.array(model.predict(x_data))                    #훈련한 모델에 x_data를 넣음

acc = accuracy_score(y_data, y_predict)
print(f"acc= {acc}")  