# 이진분류, 다중 분류

#12 Copy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error , accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import time
from keras.callbacks import EarlyStopping   

#1. 데이터
datasets = load_breast_cancer() #데이터들이 datasets 변수에 들어감
print(datasets)                     
print(datasets.DESCR)
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
# 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target

print(x)    #(569, 30)
print(y)    #(569,)
print(x.shape, y.shape)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([212, 357]))
x_train, x_test, y_train ,y_test = train_test_split(
                                                    x, y,
                                                    train_size=0.75 ,
                                                    random_state=139,
                                                    shuffle=True,
                                                ) 

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)



#2. 모델
model = Sequential()
model.add(Dense(40,activation='relu',input_shape=(30, )))
model.add(Dense(80,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#relu는 모든 값을 0이상으로 보내기때문에 마지막에서 사용하면 안됨   

#3. 컴파일, 훈련
#이진 분류는 무조건 binary_crossentropy / sigmoid 
model.compile(loss = "binary_crossentropy", optimizer="adam",
              metrics=['accuracy'], )
start_time = time.time()

es = EarlyStopping( 
    monitor='val_loss',  #멈추는 값               
    mode = 'min',    #멈추는 기준
    patience = 100,   #멈추는 횟수
    restore_best_weights=True, #멈춘다면 값들중 가장좋은 가중치값을 쓰겠다.
    )

model.fit(x_train, y_train, 
          epochs=30, batch_size=16 , 
          validation_split=0.2, callbacks= [es]
          ) 

ens_time = time.time()

print(f"걸린시간 : {round(ens_time - start_time,2)}")

#4 평가, 예측

loss = model.evaluate(x_test,y_test)    #loss = [0.21378009021282196, 0.9160839319229126]
print(f"loss = {loss}")

y_predict = np.round(model.predict(x_test),2) #오류나기떄문에 round 사용 0에서 1사이의 수치 데이터
print(f"y_presict(0~1사이) : {y_predict}")  

acc_score = accuracy_score(y_test, y_predict)
print(f"accuracy_score: {acc_score}")
    