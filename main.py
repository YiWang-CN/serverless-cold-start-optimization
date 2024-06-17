'''
测试使用，无意义
'''


# from sklearn.preprocessing import MinMaxScaler
# import torch
# import numpy as np

# # 从列表创建 NumPy 数组
# my_list = [1, 2, 3, 4]
# my_array = np.array(my_list)
# print(my_array)

# # 创建MinMaxScaler对象
# scaler = MinMaxScaler(feature_range=(0, 1))

# # 训练数据（假设是一列数据）
# train_data = [[1], [2], [3], [4]]
# test_data = [[5],[6],[7],[8]]
# # 学习缩放参数并进行缩放
# train_data_scaled = scaler.fit_transform(train_data)
# # train_data_scaled = scaler.transform(train_data)
# test_data = scaler.transform(test_data)
# test_data = torch.FloatTensor(test_data).view(-1)

# tensor = torch.randn(1, 3)
# tensor = torch.FloatTensor(tensor).view(-1)


# # 打印缩放后的数据
# print(train_data_scaled)
# print(test_data)
# print(tensor)



# my_dict = {'a': 1, 'b': 2, 'c': 3}
# new_dict = my_dict.copy()

# # 在迭代过程中修改新字典
# for key in new_dict:
#     if key == 'b':
#         my_dict['new_key'] = 4  # 在迭代中添加新键值对
# print(my_dict)
# print(new_dict)

dict1 = {}
set1 = set()
print(type(dict1))
print(type(set1))