##캐글 풀이

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes   #교육용데이터
import pandas as pd

#1 데이터
path = './_data/'

train_csv = pd.read_csv(path + "train.csv", index_col=0 ) #'./_data/train.csv'
test_csv = pd.read_csv(path + "test.csv", index_col=0 ) #0번째 컬럼(날짜) 인덱스(데이터취급x)함
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0 ) #0번째 컬럼 인덱스(데이터취급x)함

x = train_csv.drop(['casual','registered','count'],axis=1) 
# print(x)

# print("--------------------------------")
y = train_csv['count'] #count 컬럼만 넣겠다


x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.75,
    random_state=310,
    shuffle=True, 
)

#가장 큰 값으로 데이터를 다 나눈다면 소수점으로 다 떨어지게 되어 0~1사이의 값으로 나오게 됨 

#2 모델
model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(8, )))
model.add(Dropout(0.3))
model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150,activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(150,activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(10,activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(1,activation='linear',))

import time
start_time = time.time()
#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')        # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값 
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

ens_time = time.time()

print(f"걸린시간 : {round(ens_time - start_time,2)}")

#4 평가 예측    
loss = model.evaluate(x_test , y_test)  #loss: 22104.080078125  #Dropout 시 loss =  23002.857421875
#evaluate에는 Dropout이 적용이 안됨
print("loss = ", loss)   

y_predict = model.predict([x_test])
#print("[x_test] 의 예측값 : ", y_predict)

from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
rmse = root_mean_squared_error(y_test, y_predict) #mse값이 너무 크니까 사용
print(" rmse : ", rmse)      #rmse :  151.66693115234375

