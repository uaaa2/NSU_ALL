import tensorflow as tsf
from keras.models import Sequential , load_model
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

x = train_csv.drop(['casual', 'registered', 'count'], axis= 1)


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


#2 모델 학습

# #3. 컴파일 , 훈련
#====================================================
model = load_model("./_save/keras23_mcp1.keras")
#==============================================


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(f"loss: {loss}")  
#loss: 22104.080078125
#loss: 22104.080078125

y_predict = model.predict(x_test)      
print(f"y_predict: {y_predict}")

