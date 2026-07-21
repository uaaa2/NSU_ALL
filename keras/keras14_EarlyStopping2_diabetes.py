from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import numpy as np
from keras.callbacks import EarlyStopping 



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

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', # 'auto'로 하면 알아서 조정 함 
    patience=100,
    restore_best_weights=True,  #
)

x_train, x_test, y_train ,y_test = train_test_split(
                                                    x, y,
                                                    train_size=0.75 ,
                                                    random_state=2,
                                                    shuffle=True,
                                                ) 




#2. 모델
model = Sequential()
model.add(Dense(10,input_shape=(10, )))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = "mse", optimizer="adam")
model.fit(x_train, y_train, 
          epochs=10999990,batch_size=16,
          validation_split=0.2, callbacks=[es]
          )

#4 평가, 예측
y_predict = model.predict(np.array(x_test))

r2 = r2_score(y_test,y_predict)  #원값과 예측값
print(f"r2 score: {r2}")  

rmse = root_mean_squared_error(y_test,y_predict)
print(f"rmse: {rmse}")

mse = mean_squared_error(y_test,y_predict)
print(f"mse: {mse}")