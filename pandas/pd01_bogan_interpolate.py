"""
결측치 처리 

#1. 삭제 - 행, 또는 열(열이 중요하지 않거나 열에 결측치가 너무 많을때 ex. 열의 90%가 결측치임)    

#2. 임의의값 (특정값)
    - 0 : fline
    - 평균 : mean (이상치 조심) 
    - 중위 : median
    - 앞값 : ffill
    - 뒷값 : bfill 
    - 특정값 : (조건 보고 넣음)
    - 기타등등 : 
   
#3. interolate / 보간 ( 알려진 데이터 점 집합의 범위 내에 새 데이터 점을 추가하는 기법) 
#4. 모델 : .predict 0(값을 예측해서), (가급적 다른 모델 사용)
#5. 부스팅 계열 모델 : 통상 이상치, 결측지에 대해 영향을 덜 받는다. 
"""
import pandas as pd 
import numpy as np

dates = [
    '24/7/2026', '25/7/2026', '26/7/2026',
    '27/7/2026', '28/7/2026', '29/7/2026'
]
dates = pd.to_datetime(dates) #시간 데이터 형태로 변형 
#시리즈, 데이터 프레임
ts = pd.Series([2, np.nan, np.nan,  8, 10, np.nan,], index=dates) #dates의 요소와 리스트안에 요소를 대치함 
print(ts)
print("="*100)
ts = ts.interpolate()
print(ts)