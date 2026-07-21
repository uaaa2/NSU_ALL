from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes  


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

# print(x_train.shape, x_test.shape) # (331, 10) (111, 10)
# print(y_train.shape, y_test.shape) # (331,) (111,)
# # exit()
#2 모델
model = Sequential()
model.add(Dense(100, input_shape=(10,)))              
model.add(Dense(120))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(370))
model.add(Dense(360))
model.add(Dense(1))

# model.summary()
# model.save("./_save/keras_22_1.save_model.keras") #상대경로방식

model.summary()

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')        # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값  
model.fit(x_train , y_train , epochs=100 , batch_size=4)                                                     
model=load_model("./_save/keras_22_1.save_model.keras")


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






