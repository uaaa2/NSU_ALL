import pandas as pd
import numpy as np

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, np.nan],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan]
])
data = data.T #열과 행을 바꿈 (5,4)
data.columns = ['x1','x2','x3','x4',]   #칼럼에 이름을 붙임
print(data)
print('#'*100)
#0번 결측치 확인
# print(data.isnull())                   #전체 표 출력 후 None칸 확인 
# print(data.isnull().sum())             #칼럼마다 None의 개수 확인 
# print(data.info())                     #칼럼마다 None이 아닌 개수 확인

#1. 결측지 삭제
#print(data.dropna())                     #None이 있는 행을 다 지운다. 3행만 출력 됨 
#print(data.dropna(axis=0))               #None이 있는 행을 다 지운다. 3행만 출력 됨 
#print(data.dropna(axis=1))               #None이 있는 열을 다 지운다. 1열만 출력 됨 

#2_1. 특정값 - 평균
# means = data.mean()                       #칼럼별 평균 출력 
# data2 = data.fillna(means)
# print(data2) 

#2_2 특정값 - 중위감
med = data['x1'].median()                       #칼럼의 중위값
# print(med)

# data3 = data.fillna(med)                  #인수에 데이터로 data를 채우는 함수
# print(data3)

#2_3 특정값 - 0             
# data4 = data.fillna(0)                      #None칸을 0.0으로 채움
# print(data4)

#2_4 특정값 - 777
# data5 = data.fillna(777)
# print(data5)

#2_5 특정값 - ffill                 
# data6 = data.ffill()                    #같은 칼럼의 위에 있는 값을 가져다 쓰겠다
# #print(data6)                           #ffill의 첫번째 행은 참고할 위에 값이 없기 떄문에 값을 채울수 없음.

# #2_6 특정값 - bfill                       
# data7 = data.bfill()                    #같은 칼럼의 아래 있는 값을 가져다 쓰겠다
# print(data7)                            #bfill의 마지막 행은 참고할 아래 값이 없기 때문에 값을 채울 수 없음

                                        #시계열, 시간 순서대로 관측한 데이터에 잘 먹힘. 앞뒤값이 크게 바뀌지 않기 때문
                                    
###################################################################################
means = data['x4'].mean()
# print(means)        

# med = data['x4'].median()
# print(med)

#실습
"""
x1 : median
x2 : ffill
x4 : means 
"""
data['x1'] = data['x1'].fillna(data['x1'].median())
#data['x1'] = data['x1'].fillna(med)

data['x2'] = data['x2'].ffill()

data['x4'] = data['x4'].fillna(data['x4'].mean())
#data['x4'] = data['x4'].fillna(means)
print(data)