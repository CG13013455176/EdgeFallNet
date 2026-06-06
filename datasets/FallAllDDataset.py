import pickle

import numpy as np

acc_ADL =[]
acc_FALL =[]
gry_ADL =[]
gry_FALL =[]

# 使用pickle模块从文件加载数据
with open('FallAllD/ACC_ADL.pkl', 'rb') as f:
    acc_ADL = pickle.load(f)
with open('FallAllD/ACC_FALL.pkl', 'rb') as f:
    acc_FALL = pickle.load(f)
with open('FallAllD/GCC_ADL.pkl', 'rb') as f:
    gry_ADL = pickle.load(f)
with open('FallAllD/GCC_FALL.pkl', 'rb') as f:
    gry_FALL = pickle.load(f)


def get_3d_list_min_max(three_d_list):
    """
    计算三维列表的最大值和最小值
    参数：three_d_list - 三维列表（格式：[ [ [x1,x2,...], [x1,x2,...] ], ... ]）
    返回：(最小值, 最大值)，若列表为空返回(None, None)
    """
    # 遍历三维列表，提取所有元素到一维列表
    all_elements = []
    for sublist1 in three_d_list:  # 第一层（样本数）
        for sublist2 in sublist1:  # 第二层（时间步）
            all_elements.extend(sublist2)  # 第三层（特征值，x/y/z轴）

    if not all_elements:  # 处理空列表情况
        return (None, None)
    return (min(all_elements), max(all_elements))


# 分别计算四个列表的极值
acc_ADL_min, acc_ADL_max = get_3d_list_min_max(acc_ADL)
acc_FALL_min, acc_FALL_max = get_3d_list_min_max(acc_FALL)
gry_ADL_min, gry_ADL_max = get_3d_list_min_max(gry_ADL)
gry_FALL_min, gry_FALL_max = get_3d_list_min_max(gry_FALL)

# 打印结果
print("=" * 50)
print("各列表的最大值和最小值：")
print("=" * 50)
print(f"acc_ADL - 最小值: {acc_ADL_min:>8}, 最大值: {acc_ADL_max:>8}")
print(f"acc_FALL - 最小值: {acc_FALL_min:>8}, 最大值: {acc_FALL_max:>8}")
print(f"gry_ADL - 最小值: {gry_ADL_min:>8}, 最大值: {gry_ADL_max:>8}")
print(f"gry_FALL - 最小值: {gry_FALL_min:>8}, 最大值: {gry_FALL_max:>8}")
print("=" * 50)

#归一化处理
acc_ADL_new = [[[round(num*0.000244, 4) for num in sublist2] for sublist2 in sublist1]
                      for sublist1 in acc_ADL]
acc_FALL_new = [[[round(num*0.000244, 4) for num in sublist2] for sublist2 in sublist1]
                      for sublist1 in acc_FALL]

gry_ADL_new = [[[round((num*0.07*8)/2000, 4) for num in sublist2] for sublist2 in sublist1]
                      for sublist1 in gry_ADL]
gry_FALL_new = [[[round((num*0.07*8)/2000, 4) for num in sublist2] for sublist2 in sublist1]
                      for sublist1 in gry_FALL]

#acc与gry融合处理
data_ADL = [[list(acc) + list(gyr) for acc, gyr in zip(acc_sublist, gyr_sublist)]
                                 for acc_sublist, gyr_sublist in zip(acc_ADL_new, gry_ADL_new)]

data_FALL = [[list(acc) + list(gyr) for acc, gyr in zip(acc_sublist, gyr_sublist)]
                                 for acc_sublist, gyr_sublist in zip(acc_FALL_new, gry_FALL_new)]

acc_ADL_new_min, acc_ADL_new_max = get_3d_list_min_max(acc_ADL_new)
acc_FALL_new_min, acc_FALL_new_max = get_3d_list_min_max(acc_FALL_new)
gry_ADL_new_min, gry_ADL_new_max = get_3d_list_min_max(gry_ADL_new)
gry_FALL_new_min, gry_FALL_new_max = get_3d_list_min_max(gry_FALL_new)

# 打印结果
print("=" * 50)
print("各列表的最大值和最小值：")
print("=" * 50)
print(f"acc_ADL - 最小值: {acc_ADL_new_min:>8}, 最大值: {acc_ADL_new_max:>8}")
print(f"acc_FALL - 最小值: {acc_FALL_new_min:>8}, 最大值: {acc_FALL_new_max:>8}")
print(f"gry_ADL - 最小值: {gry_ADL_new_min:>8}, 最大值: {gry_ADL_new_max:>8}")
print(f"gry_FALL - 最小值: {gry_FALL_new_min:>8}, 最大值: {gry_FALL_new_max:>8}")
print("=" * 50)

x = []  # 存储交替的二维列表
y = []  # 存储来源编号（0或1）
# 处理data_ADL中比data_FALL多出的部分
len_adl = len(data_ADL)
len_fall = len(data_FALL)
for i in range(len_fall, len_adl):
    x.append(data_ADL[i])
    y.append(0)

# 交替存储剩余的二维列表，并添加来源编号
index_adl = 0
index_fall = 0
while index_adl < len_adl and index_fall < len_fall:
    x.append(data_ADL[index_adl])
    y.append(0)
    index_adl += 1

    if index_fall < len_fall:
        x.append(data_FALL[index_fall])
        y.append(1)
        index_fall += 1

labels=['ADL','FALL'] # labels : ADL=0; Fall=1

def check_same_shape(three_d_list):
    # 首先检查列表是否为空
    if not three_d_list:
        return True

        # 获取第一个二维列表的形状作为参考
    ref_shape = (len(three_d_list[0]), len(three_d_list[0][0])) if three_d_list[0] and three_d_list[0][0] else (0, 0)

    # 遍历每个二维列表，检查其形状是否与参考形状相同
    for sublist in three_d_list:
        # 如果子列表的长度不等于参考长度，返回False
        if len(sublist) != ref_shape[0]:
            return False
            # 如果子列表中的任一元素的长度不等于参考长度，返回False
        for element in sublist:
            if len(element) != ref_shape[1]:
                return False

                # 如果所有二维列表的形状都相同，返回True
    return True

print(check_same_shape(x))

with open('FallAllD/FallAllD_Data1.pkl', 'wb') as f:
    pickle.dump(x, f)
with open('FallAllD/FallAllD_lable1.pkl', 'wb') as f:
    pickle.dump(y, f)

