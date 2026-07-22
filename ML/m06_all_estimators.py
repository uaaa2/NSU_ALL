import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators

#1. 데이터 
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=123, test_size= 0.2 
)
sclaer = RobustScaler()
x_train = sclaer.fit_transform(x_train)
x_test = sclaer.transform(x_test)

#2. 모델구성
#model = RandomForestRegressor()
all_models = all_estimators(type_filter='regressor')

# print("all_models", all_models)
# print("모델의 갯수:", len(all_models))


for (name, algorithm) in all_models:
    try: 
        model = algorithm()
        
        #3. 훈련 
        model.fit(x_train,y_train)
        #4, 평가, 예측
        results = model.score(x_test, y_test)
        print(name,'의 정답률',results)
    except: 
        print(name,'은(는) 에러뜬 분!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')