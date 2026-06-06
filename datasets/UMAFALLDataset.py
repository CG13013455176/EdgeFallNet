import numpy as np
import pickle

x=[]
y=[]

file_ADL="UMAFALL/ADL.txt"# contains all the ADLs data windows (length of the window 401 * 3)
file_FALL="UMAFALL/FALL.txt"# contains all the Falls data windows (length of the window 401 * 3)
data_ADL = np.loadtxt(file_ADL, dtype=float, delimiter=',')
data_FALL = np.loadtxt(file_FALL, dtype=float, delimiter=',')

print(len(data_ADL))
print(len(data_FALL))

for i in range(len(data_FALL),len(data_ADL)):
        x.append(data_ADL[i])
        y.append(0)

for i in range(len(data_FALL)):
    x.append(data_ADL[i])
    y.append(0)
    x.append(data_FALL[i])
    y.append(1)

x1 = np.array(x)
x2 = np.round(x1 / 8, 4)

# 将y也转换为numpy数组
y_array = np.array(y)

# 分别保存x2和y为pkl文件
with open('UMAFALL/UMAFALL_Data.pkl', 'wb') as f:
    pickle.dump(x2, f)

with open('UMAFALL/UMAFALL_lable.pkl', 'wb') as f:
    pickle.dump(y_array, f)

print("x2和y已分别保存为UMAFALL_Data.pkl和UMAFALL_lable.pkl文件")