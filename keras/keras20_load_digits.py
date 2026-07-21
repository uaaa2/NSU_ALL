from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd 
import time
from keras.callbacks import EarlyStopping   

datasets = load_digits()
print(datasets)           
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)  #(1797, 64)
# print(y.shape)  #(1797, )

print(np.unique(y, return_counts=True)) 
exit()

y = pd.get_dummies(y)   #one hot 인코딩
print(y)



x_train, x_test, y_train ,y_test = train_test_split(
                                                    x, y,
                                                    train_size=0.75 ,
                                                    random_state=1,
                                                    shuffle=True,
                                                ) 


model = Sequential()
model.add(Dense(40,activation='relu',input_shape=(64, )))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'], )

es = EarlyStopping( 
    monitor='val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights=True,
    )


model.fit(x_train, y_train, 
          epochs=100, batch_size=8 , 
          validation_split=0.2, callbacks= [es]
          )     


loss = model.evaluate(x_test,y_test)   
print(f"loss: {loss}")

y_predict = model.predict(x_test)

y_predict_class = np.argmax(y_predict, axis=1)
y_test_class = np.argmax(y_test.values, axis=1)


acc_score = accuracy_score(y_test_class, y_predict_class)
print(f"accuracy_score: {acc_score}")