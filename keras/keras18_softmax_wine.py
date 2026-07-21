from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error , accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine
import numpy as np
import pandas as pd 
import time
from keras.callbacks import EarlyStopping   

datasets = load_wine()
print(datasets)           
print(datasets.DESCR)
print(datasets.feature_names)
exit()

x = datasets.data
print(x.shape)  #(178, 13)

y = datasets.target
print(y.shape)  #(178,)

print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48]))

#클래스 3개 컬럼 13개
y = pd.get_dummies(y)
print(y)


x_train, x_test, y_train ,y_test = train_test_split(
                                                    x, y,
                                                    train_size=0.75 ,
                                                    random_state=139,
                                                    shuffle=True,
                                                ) 


model = Sequential()
model.add(Dense(40,activation='relu',input_shape=(13, )))
model.add(Dense(80,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'], )

es = EarlyStopping( 
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
    )

start_time = time.time()

model.fit(x_train, y_train, 
          epochs=100, batch_size=8 , 
          validation_split=0.2, callbacks= [es]
          ) 

ens_time = time.time()

print(f"걸린시간 : {round(ens_time - start_time,2)}")

loss = model.evaluate(x_test,y_test)   
print(f"loss = {loss}")

y_predict = model.predict(x_test)
print(f"y_presict: {y_predict}")  



y_predict_class = np.argmax(y_predict, axis=1)
y_test_class = np.argmax(y_test.values, axis=1)

print(f"y_test_class:{y_test_class}")
print(f"y_predict_class:{y_predict_class}")

acc_score = accuracy_score(y_test_class, y_predict_class)
print(f"accuracy_score: {acc_score}")