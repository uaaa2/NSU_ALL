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

#0번 결측치 확인
print(data.isnull()) #None칸 확인 
print(data.isnull().sum())
