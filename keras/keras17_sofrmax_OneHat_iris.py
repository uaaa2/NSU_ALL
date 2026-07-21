from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error , accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
import numpy as np
import pandas as pd 
import time
from keras.callbacks import EarlyStopping   

datasets = load_iris()
print(datasets)           
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
print(x)

y = datasets.target
print(y)

print(np.unique(y, return_counts=True))

y = pd.get_dummies(y)
print(y)


x_train, x_test, y_train ,y_test = train_test_split(
                                                    x, y,
                                                    train_size=0.75 ,
                                                    random_state=139,
                                                    shuffle=True,
                                                ) 


model = Sequential()
model.add(Dense(40,activation='relu',input_shape=(4, )))
model.add(Dense(80,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'], )
start_time = time.time()

es = EarlyStopping( 
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
    )

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

print(f"y_test_class:{y_test_class[:5]}")
print(f"y_predict_class:{y_predict_class}")
exit()

acc_score = accuracy_score(y_test_class, y_predict_class)
print(f"accuracy_score: {acc_score}")