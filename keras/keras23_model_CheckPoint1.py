import tensorflow as tsf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import numpy as np
import pandas as pd
import time 

#중간에 튀는 데이터를 이상치, 표 중간에 빈 칸을 결측치

#1. 데이터 
path = "./_data/"

train_csv =  pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv",index_col=0)
submit_csv = pd.read_csv(path + "sampleSubmission.csv",index_col=0)

#csv파일을 읽는다. 첫번째 컬럼을 인덱스로 취급하겠다. 기존 인덱스 번호는 없어짐

x = train_csv.drop(['casual', 'registered', 'count'], axis= 1)
# season  holiday  workingday  weather  temp  atemp  humidity  windspeed 

y = train_csv['count']  #count 컬럼만 넣음 
print(y)    #Name: count, Length: 10886, dtype: int64
print(y.shape)  #(10886,)

x_train, x_test, y_train, y_test = train_test_split (
                                x, y,
                                train_size=0.75,
                                test_size=0.25,
                                shuffle=True,
                                random_state= 371
)

#



#로그 사용 RMSLE 대신 유사 지표 RMSE 사용
#2 모델 학습
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(8, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

#activation 활성화 함수
#activation= 'relu' 음수부분을 없앤다.
#activation= 'linear' defult 

#3. 컴파일 , 훈련
model.compile(loss = "mse", optimizer="adam")
start_time = time.time()

from keras.callbacks import EarlyStopping , ModelCheckpoint

es = EarlyStopping( 
    monitor='val_loss',  #멈추는 값               
    mode = 'min',    #멈추는 기준
    verbose=1,
    patience = 100,   #멈추는 횟수
    
    restore_best_weights=True, #멈춘다면 값들중 가장좋은 가중치값을 쓰겠다.
    )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath="./_save/keras23_mcp1.keras"
)
model.fit(x_train, y_train, 
          epochs=21231235, batch_size=32 , 
          validation_split=0.2, 
          callbacks= [es,mcp]
          ) 
#loss값이 낮아져도 train data안에서 실재로 개선중인지 과적합중인지 알 수 없음. 
#때문에 validation으로 train에 n%를 빼 훈련 시키지않은 데이터로 검증함 
#이 검증 데이터의 loss값을 val_loss값이라고 함
#로컬 미니마 글로벌 미니마 

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(f"loss: {loss}")

y_predict = model.predict(x_test)      
print(f"y_predict: {y_predict}")

rmse=root_mean_squared_error(y_test,y_predict)
print(f"rmse: {rmse}")

################ CSV 파일 만들기 ###################

y_submit = model.predict(test_csv)  # #output_dim = 1, x_test에는 없는 count 칼럼의 값을 예측하여 y_predict에 넣음
print(y_submit)

submit_csv['count'] = y_submit      #count 칼럼만 있는 y_submit값들을 datetime값과 count 칼럼만 있는 subnit_csv에 대입 
print(submit_csv)

# submit_csv.to_csv(path + "sunmission_0717_1453.csv") #"C:\NSU_all\_data\sunmission_0717_.csv" 파일 생성
# print(f"걸린시간 : {round(end_time-start_time,2)} 초") #소수점 두번째 자리에서 자름