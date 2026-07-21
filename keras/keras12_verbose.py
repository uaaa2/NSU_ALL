

#11_2 Copy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import numpy as np
import time

#1. 데이터
datasets = load_diabetes() #데이터들이 datasets 변수에 들어감
print(datasets)                     
print(datasets.DESCR)
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
# 'Population', 'AveOccup', 'Latitude', 'Longitude']
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

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
model.add(Dense(40,input_shape=(10, )))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = "mse", optimizer="adam")
start_time = time.time()
model.fit(x_train, y_train, epochs=100,batch_size=32, verbose= 4)   #기존 3.75초 / 
#verbose 사용시 인터페이스에 진행사황을 보여주지 않아 딜레이가 생기지 않음 3.67 defult: verbose = 1
#verbose =2 / 진행바없이 텍스트만 출력 verbose = 3 epoch 만 나옴
ens_time = time.time()

print(f"걸린시간 : {round(ens_time - start_time,2)}")

#4 평가, 예측
y_predict = model.predict(np.array(x_test))

r2 = r2_score(y_test,y_predict)  #원값과 예측값
print(f"r2 score: {r2}")
  
rmse = root_mean_squared_error(y_test,y_predict)
print(f"rmse: {rmse}")

mse = mean_squared_error(y_test,y_predict)
print(f"mse: {mse}")