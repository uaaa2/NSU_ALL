
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes   #교육용데이터
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np


#1 데이터
datasets = load_diabetes()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names) 

x = datasets.data
y = datasets.target
# print(x)
# print(y)

print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.90,
    random_state=310,
    shuffle=True, 
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2 모델
model = Sequential()
model.add(Dense(100, input_shape=(10,)))              
model.add(Dense(120))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')       
model.fit(x_train , y_train , epochs=100 , batch_size=4)                                                     



#4. 평가, 예측
loss = model.evaluate(x_test , y_test)  #stand: loss =  1735.9307861328125 
print("loss = ", loss)                  #Minmax: loss =  1843.2406005859375

model.predict(x_test) 

y_predict  = model.predict([x_test]) 

r2 = r2_score(y_test, y_predict) #원값, 예측값 - r2스코어 계산
print("r2_score : ", r2) # 1에 가까울수록 좋음







