import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50,20])

def outlier(data) :
    quatlie_1, q2 , quarylie_3 = np.percentile(data, [25, 50, 75])
    #각각1사분위, 중앙 , 3사분위 값을 넣음 
    print("1사분위:", quatlie_1) #4.0
    print("q2:", q2)            #중위점, 2사분위 #7.0
    print("3사분위:", quarylie_3)#3사분위 #10.0
    iqr = quarylie_3 - quatlie_1 #중점이 되는 데이터의 값 6.0
    #1사분위 - IQR*1.5 / 3사분위 +  * 1.5 해서 정상 범위를 정의 함  정상범위: -5.0 19.0
    print("IQR:", iqr)
    lower_bound = quatlie_1 - (iqr * 1.5)   #정상 데이터 범위 최소 수치: -5
    upper_bound = quarylie_3 + (iqr * 1.5)  #정상 데이터 범위 최대 수치: 19
    print("범위:",lower_bound, upper_bound)
    return np.where((data > upper_bound) | (data < lower_bound)),   \
        iqr, lower_bound, upper_bound
    #이상치 값들을 where안에 다 묶어서 outlier_loc에 반환시키고 마지막 세개는 중위값, 1사분위, 3사분위
outlier_loc, iqr, low, up = outlier(aaa) #aaa리스트의 정상 범위 / 중위값 / 1사분위 / 3사분위 값 들어감
print('이상치의 위치:', outlier_loc)

import matplotlib.pyplot as plt

plt.boxplot(aaa)    #aaa의 전체 범위 출력 
plt.axhline(up, color='red', label='upper bound')
plt.axhline(low, color='blue', label='lower_bound')
plt.legend()
plt.show()