
#보스턴: 스킷런에 기본 중 기본..이었는데 인종이슈로 못씀
#대안으로 나온게 california 데이터 셋

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing   #교육용데이터


#1 데이터
datasets = fetch_california_housing()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names) #attribute = 8 , input_dim= 8
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
# 'Population', 'AveOccup', 'Latitude', 'Longitude']


x = datasets.data
y = datasets.target
# print(x)
# print(y)

#print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.75,
    random_state=333,
    shuffle=True,
    
)

# print(x_train.shape, x_test.shape) # (15480, 8) (5160, 8)
# print(y_train.shape, y_test.shape) # (15480,) (5160,)

#2 모델
model = Sequential()
model.add(Dense(110, input_shape=(8,)))              
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')        # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값  
model.fit(x_train , y_train , epochs=1000,  )                                                     


#4. 평가, 예측
loss = model.evaluate(x_test , y_test)
print("loss = ", loss)   

model.predict(x_test) #최종예측

y_predict  = model.predict([x_test]) # 100번째 w를 이용해서 예측값 = wx+b
# print("y_test의 원값 : ", y_test)
# print("[x_test] 의 예측값 : ", y_predict)

#평가
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
r2 = r2_score(y_test, y_predict) #원값, 예측값 - r2스코어 계산
print("r2_score : ", r2) # 1에 가까울수록 좋음

rmse = root_mean_squared_error(y_test, y_predict) #mse값이 너무 크니까 사용
print(" rmse : ", rmse) 

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)


"""
train_size=0.75, 
model.add(Dense(90, input_shape=(8,)))              
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1)) 
epochs=180 -> r2= 0.5216640219273653

train_size=0.80
model.add(Dense(90, input_shape=(8,)))              
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))
epochs=180
r2_score :  0.5211776931740326

epochs=180, batch_size=28
r2_score :  0.5256446381508508
"""
