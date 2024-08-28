# import numpy as np
# import tvm
# from tvm import  tir

# rlt = tir.abs(-100)
# print("abs(-100) = %d" % rlt)

import h5py  
  
# 打开.h5文件  
with h5py.File('model.h5', 'r') as f:  
    # 遍历文件的根组  
    for name in f:  
        print(name)  
        # 根据需要继续遍历子组或读取数据集  
        # 例如，打印某个数据集的内容  
        # dataset = f[name]  
        # print(dataset[:])