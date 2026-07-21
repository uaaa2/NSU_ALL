#캐글 풀이

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes   #교육용데이터
import pandas as pd

#1 데이터
path = './_data/'

train_csv = pd.read_csv(path + "train.csv", index_col=0 ) #'./_data/train.csv'
test_csv = pd.read_csv(path + "test.csv", index_col=0 ) #0번째 컬럼(날짜) 인덱스(데이터취급x)함
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0 ) #0번째 컬럼 인덱스(데이터취급x)함
#print(train_csv)
#print(train_csv.shape) #(10886, 11)
#print(test_csv)
# print(test_csv.shape) #(6493, 8)
# print(submit_csv)
# print(submit_csv.shape) # (6493, 1)
# print(train_csv.columns) 
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='str')
# exit()
#레지스트,캐쥬얼,카운트 는 복잡하기 때문에 분리
x = train_csv.drop(['casual','registered','count'],axis=1) #열을 뺀다 #axis= 축 어디를?
# print(x)

# print("--------------------------------")
y = train_csv['count'] #count 컬럼만 넣겠다
# print(y)
# print(y.shape) #(10886,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.75,
    random_state=310,
    shuffle=True, 
)



#2 모델
model = Sequential()
model.add(Dense(100,input_shape=(8, )))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(100))
model.add(Dense(1))

import time
start_time = time.time()
#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')        # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값  
model.fit(x_train , y_train , epochs=200 ) 

ens_time = time.time()

print(f"걸린시간 : {round(ens_time - start_time,2)}")

#4 평가 예측
loss = model.evaluate(x_test , y_test)
print("loss = ", loss)   

y_predict = model.predict([x_test])
#print("[x_test] 의 예측값 : ", y_predict)


from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
rmse = root_mean_squared_error(y_test, y_predict) #mse값이 너무 크니까 사용
print(" rmse : ", rmse) 

